import cv2
import numpy as np
import pandas as pd
import json
from utils._fbx_utils_ import generateFBX

from utils.smplx_utils import smplx_parents_index_list, smplx_joint_names, get_verts, process_json_animation_file
from utils.smplx_utils import smooth_animation_data, blend_animations, prepare_weights_data

import glob
import os
import argparse
# ================= 加载模型与计算 T-Pose =================
SMPLX_MODEL_PATH = r"SMPLest_X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz"
model_data = np.load(SMPLX_MODEL_PATH, allow_pickle=True)

weights = model_data['weights']   # shape为(10475, 55)
faces_mesh = model_data['f']              # shape为(20908, 3)

num_joints = 55
full_pose = np.zeros((num_joints, 3))
betas = np.zeros(10)

# 计算 T-Pose 下的关节位置，用于提取骨骼初始长度(偏移)
verts3D, j_t_pose = get_verts(model_data, full_pose.flatten(), betas)
verts3D = verts3D * 200
j_t_pose = j_t_pose * 200
# 计算每个关节的本地坐标
j_t_pose_local = []
for i, name in enumerate(smplx_joint_names):
    bone_parent_index = smplx_parents_index_list[i]
    if i == 0:
        trans = j_t_pose[i]
    else:
        trans = j_t_pose[i] - j_t_pose[bone_parent_index]
    j_t_pose_local.append(trans)



# 解析命令行参数
parser = argparse.ArgumentParser(description='Generate Maya Action')
parser.add_argument('--input_paths', type=str, default='video_clips', help='输入视频路径')
parser.add_argument('--output_paths', type=str, default='video_clips', help='输出文件路径')
args = parser.parse_args()

idle_pose_path = r"idle_pose\idle_pose.json"
with open(idle_pose_path, 'r') as f:
    idle_pose_smplx_params_list = json.load(f)
num_frames = len(idle_pose_smplx_params_list)
# idle pose 有三百帧
idle_animation_data = process_json_animation_file(idle_pose_smplx_params_list, num_frames, eular=True)

input_paths = args.input_paths
smplx_params_files_path = glob.glob(os.path.join(input_paths, "*.json"))

for file_path in smplx_params_files_path:
    fbx_file_path = os.path.join(args.output_paths, os.path.basename(file_path).replace(".json", ".fbx"))
    os.makedirs(args.output_paths, exist_ok=True)
    with open(file_path, 'r') as f:
        smplx_params_list = json.load(f)
    num_frames = len(smplx_params_list)
    
    new_animation_data = process_json_animation_file(smplx_params_list, num_frames, eular=True, upper_body=True)
    new_animation_data = smooth_animation_data(new_animation_data, window_size=5)
    new_animation_data = blend_animations(new_animation_data, idle_animation_data, num_frames)

    weights_dict = prepare_weights_data(weights)
    mesh_world_matrix = np.eye(4, dtype=np.float64) 
    geometry = (verts3D, faces_mesh, weights_dict, mesh_world_matrix, j_t_pose, j_t_pose_local)
    generateFBX(fbx_file_path, new_animation_data, geometry)

    print("SMPLX FBX 生成完毕！")