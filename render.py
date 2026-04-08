import os
# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import tqdm
import glob
import json
import numpy as np
import cv2
import torch
import smplx
import pyrender
import trimesh
from utils.smplx_utils import smooth_animation_data, blend_animations, prepare_weights_data, process_json_animation_file

SMPLX_MODEL_PATH = r"SMPLest_X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz"
RENDER_RESOLUTION = 512


# --- 3. 加载SMPLX模型 ---
# 使用GPU如果可用，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smplx_model = smplx.create(
    SMPLX_MODEL_PATH, 
    model_type='smplx', 
    gender='neutral', 
    use_pca=False, 
    # use_face_contour=True,
    use_face_contour=False,
    num_betas=10,
    flat_hand_mean=True,
).to(device)

def compute_look_at_matrix(eye, target, up=np.array([0.0, -1.0, 0.0])):
    """纯numpy计算的look_at矩阵，符合OpenGL/Pyrender标准"""
    # 计算三个正交基向量
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    true_up = np.cross(right, forward)
    
    # 构建 4x4 矩阵
    pose = np.eye(4)
    # 注意：Pyrender 相机默认朝向 -Z，所以 forward 要取反放在第三行
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose

def init_renderer():
    renderer = pyrender.OffscreenRenderer(RENDER_RESOLUTION, RENDER_RESOLUTION)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    
    # 使用我们自己的函数生成矩阵
    camera_pose = compute_look_at_matrix(
        eye=np.array([0.0, -0.4, -2.0]), 
        target=np.array([0.0, -0.4, 0.0])
    )

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    
    return renderer, scene

def render_frame(animation_data, renderer, scene, mesh_node_handle=[None]):
    """
    输入: 
    animation_data: 动画数据列表，每个元素是一个帧的动画数据（欧拉角格式）
    输出: 512x512 的 RGB 图像 (numpy array)
    """
    # 提取欧拉角
    global_orient = animation_data[0:1]
    body_pose = np.array(animation_data[1:22], dtype=np.float32).reshape(21, 3)
    jaw_pose = np.array(animation_data[22:23], dtype=np.float32).reshape(1, 3)
    eye_pose = np.array(animation_data[23:25], dtype=np.float32)
    left_hand_pose = np.array(animation_data[25:40], dtype=np.float32).reshape(15, 3)
    right_hand_pose = np.array(animation_data[40:55], dtype=np.float32).reshape(15, 3)

    global_orient = euler_to_axis_angle(global_orient)
    body_pose = euler_to_axis_angle(body_pose)
    left_hand_pose = euler_to_axis_angle(left_hand_pose)
    right_hand_pose = euler_to_axis_angle(right_hand_pose)
    jaw_pose = euler_to_axis_angle(jaw_pose)

    global_orient = np.array(global_orient, dtype=np.float32).reshape(1, 3)
    body_pose = np.array(body_pose, dtype=np.float32).reshape(1, 63)
    left_hand_pose = np.array(left_hand_pose, dtype=np.float32).reshape(1, 45)
    right_hand_pose = np.array(right_hand_pose, dtype=np.float32).reshape(1, 45)
    jaw_pose = np.array(jaw_pose, dtype=np.float32).reshape(1, 3)

    # 将数据转为Tensor送入SMPLX
    with torch.no_grad():
        body_output = smplx_model(
            global_orient=torch.tensor(global_orient, device=device),
            body_pose=torch.tensor(body_pose, device=device),
            left_hand_pose=torch.tensor(left_hand_pose, device=device),
            right_hand_pose=torch.tensor(right_hand_pose, device=device),
            jaw_pose=torch.tensor(jaw_pose, device=device),
        )
        
    vertices = body_output.vertices.detach().cpu().numpy().squeeze()
    faces = smplx_model.faces
    
    # 1. 定义肤色并生成顶点颜色数组
    skin_color = np.array([0.75, 0.6, 0.55, 1.0])
    vertex_colors = np.tile(skin_color, (len(vertices), 1))
    
    # 2. 创建包含顶点颜色的 trimesh
    body_mesh = trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        vertex_colors=vertex_colors,
        process=False
    )
    
    # 3. 转换为 pyrender mesh (此时不会再报错)
    pr_mesh = pyrender.Mesh.from_trimesh(body_mesh)
    
    # 移除上一帧的 mesh 节点
    if mesh_node_handle[0] is not None:
        scene.remove_node(mesh_node_handle[0])
        
    mesh_node_handle[0] = scene.add(pr_mesh)
    
    # 渲染
    color, _ = renderer.render(scene)
    
    # pyrender 返回的是 RGBA，转为 RGB
    return color[:, :, :3]

def EulurAngle(rot_mat):
    """
    将旋转矩阵转换为欧拉角 (X-Y-Z顺序)
    """
    sy = np.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        y = np.arctan2(-rot_mat[2, 0], sy)
        z = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        x = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        y = np.arctan2(-rot_mat[2, 0], sy)
        z = 0
    
    # 转换为角度
    return np.array([x, y, z]) * 180 / np.pi

def euler_to_axis_angle(euler):
    """
    将欧拉角转换为轴角
    支持单个欧拉角 (3,) 或批量欧拉角 (N, 3) 输入
    """
    # 检查输入形状
    if len(euler.shape) == 1:
        # 单个欧拉角
        # 转换为弧度
        euler_rad = euler * np.pi / 180
        
        # 创建旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(euler_rad[0]), -np.sin(euler_rad[0])],
            [0, np.sin(euler_rad[0]), np.cos(euler_rad[0])]
        ])
        
        R_y = np.array([
            [np.cos(euler_rad[1]), 0, np.sin(euler_rad[1])],
            [0, 1, 0],
            [-np.sin(euler_rad[1]), 0, np.cos(euler_rad[1])]
        ])
        
        R_z = np.array([
            [np.cos(euler_rad[2]), -np.sin(euler_rad[2]), 0],
            [np.sin(euler_rad[2]), np.cos(euler_rad[2]), 0],
            [0, 0, 1]
        ])
        
        R = R_z @ R_y @ R_x
        
        # 将旋转矩阵转换为轴角
        axis, angle = cv2.Rodrigues(R)
        return axis.flatten()
    elif len(euler.shape) == 2:
        # 批量欧拉角
        axis_angles = []
        for e in euler:
            # 转换为弧度
            euler_rad = e * np.pi / 180
            
            # 创建旋转矩阵
            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(euler_rad[0]), -np.sin(euler_rad[0])],
                [0, np.sin(euler_rad[0]), np.cos(euler_rad[0])]
            ])
            
            R_y = np.array([
                [np.cos(euler_rad[1]), 0, np.sin(euler_rad[1])],
                [0, 1, 0],
                [-np.sin(euler_rad[1]), 0, np.cos(euler_rad[1])]
            ])
            
            R_z = np.array([
                [np.cos(euler_rad[2]), -np.sin(euler_rad[2]), 0],
                [np.sin(euler_rad[2]), np.cos(euler_rad[2]), 0],
                [0, 0, 1]
            ])
            
            R = R_z @ R_y @ R_x
            
            # 将旋转矩阵转换为轴角
            axis, angle = cv2.Rodrigues(R)
            axis_angles.append(axis.flatten())
        return np.array(axis_angles)
    else:
        raise ValueError("输入欧拉角的形状必须是 (3,) 或 (N, 3)")

def smooth_smplx_params(params_list, window_size=5):
    """
    对SMPLX参数进行帧间平滑处理
    1. 将轴角转换为欧拉角
    2. 处理角度连续性问题
    3. 对欧拉角进行高斯加权平滑
    4. 将平滑后的欧拉角转换回轴角
    """
    frame_num = len(params_list)
    
    # 1. 预处理：将所有轴角转换为欧拉角并处理连续性
    euler_data = []
    for frame_idx in range(frame_num):
        frame = params_list[frame_idx]
        euler_frame = {
            'frame': frame['frame'],
            'smplx_root_pose': [],
            'smplx_body_pose': [],
            'smplx_lhand_pose': [],
            'smplx_rhand_pose': [],
            'smplx_jaw_pose': [],
            'smplx_shape': frame['smplx_shape'],
            'smplx_expr': frame['smplx_expr'],
            'cam_trans': frame['cam_trans']
        }
        
        # 处理需要转换的参数
        pose_keys = ['smplx_root_pose', 'smplx_body_pose', 'smplx_lhand_pose', 'smplx_rhand_pose', 'smplx_jaw_pose']
        
        for key in pose_keys:
            pose_data = frame[key]
            euler_pose = []
            
            # 每个关节的轴角数据
            if key == 'smplx_root_pose' or key == 'smplx_jaw_pose':
                # 单个关节，3维
                bone_eular = np.array(pose_data)
                if np.linalg.norm(bone_eular) < 1e-6:
                    euler = np.array([0.0, 0.0, 0.0])
                else:
                    rot_mat, _ = cv2.Rodrigues(bone_eular)
                    euler = EulurAngle(rot_mat)
                
                # 处理角度连续性
                if frame_idx > 0:
                    prev_euler = euler_data[frame_idx-1][key][0]
                    for i in range(3):
                        diff = euler[i] - prev_euler[i]
                        if diff > 180:
                            euler[i] -= 360
                        elif diff < -180:
                            euler[i] += 360
                
                euler_pose.append(euler.tolist())
            else:
                # 多个关节，每个关节3维
                num_joints = len(pose_data) // 3
                for j_idx in range(num_joints):
                    bone_eular = np.array(pose_data[j_idx*3:(j_idx+1)*3])
                    if np.linalg.norm(bone_eular) < 1e-6:
                        euler = np.array([0.0, 0.0, 0.0])
                    else:
                        rot_mat, _ = cv2.Rodrigues(bone_eular)
                        euler = EulurAngle(rot_mat)
                    
                    # 处理角度连续性
                    if frame_idx > 0:
                        prev_euler = euler_data[frame_idx-1][key][j_idx]
                        for i in range(3):
                            diff = euler[i] - prev_euler[i]
                            if diff > 180:
                                euler[i] -= 360
                            elif diff < -180:
                                euler[i] += 360
                    
                    euler_pose.append(euler.tolist())
            
            euler_frame[key] = euler_pose
        
        euler_data.append(euler_frame)
    
    # 2. 对欧拉角进行高斯加权平滑处理
    smoothed_euler_data = []
    for i in range(frame_num):
        # 确定窗口范围
        start = max(0, i - window_size)
        end = min(frame_num, i + window_size + 1)
        
        # 收集窗口内的所有帧
        window_frames = euler_data[start:end]
        window_size_actual = len(window_frames)
        
        # 创建高斯权重
        weights = []
        for k in range(window_size_actual):
            distance = abs(k - (i - start))
            weight = np.exp(-0.5 * (distance / (window_size/2))**2)
            weights.append(weight)
        # 归一化权重
        weights = np.array(weights) / sum(weights)
        
        # 创建新的平滑后的帧
        smoothed_frame = {
            'frame': euler_data[i]['frame'],
            'smplx_root_pose': [],
            'smplx_body_pose': [],
            'smplx_lhand_pose': [],
            'smplx_rhand_pose': [],
            'smplx_jaw_pose': [],
            'smplx_shape': [],
            'smplx_expr': [],
            'cam_trans': []
        }
        
        # 对每个参数进行高斯加权平滑
        param_keys = ['smplx_root_pose', 'smplx_body_pose', 'smplx_lhand_pose', 
                     'smplx_rhand_pose', 'smplx_jaw_pose', 'smplx_shape', 'smplx_expr', 'cam_trans']
        
        for key in param_keys:
            if key in ['smplx_shape', 'smplx_expr', 'cam_trans']:
                # 对这些参数进行高斯加权平滑
                values = [frame[key] for frame in window_frames]
                values_array = np.array(values)
                # 应用权重
                weighted_values = values_array * weights[:, np.newaxis]
                smoothed_values = np.sum(weighted_values, axis=0).tolist()
                smoothed_frame[key] = smoothed_values
            else:
                # 对欧拉角进行高斯加权平滑
                num_joints = len(euler_data[i][key])
                for j in range(num_joints):
                    joint_values = []
                    for k, frame in enumerate(window_frames):
                        joint_values.append(np.array(frame[key][j]) * weights[k])
                    joint_array = np.array(joint_values)
                    smoothed_joint = np.sum(joint_array, axis=0).tolist()
                    smoothed_frame[key].append(smoothed_joint)
        
        smoothed_euler_data.append(smoothed_frame)
    
    # 3. 将平滑后的欧拉角转换回轴角
    smoothed_params = []
    for frame in smoothed_euler_data:
        axis_angle_frame = {
            'frame': frame['frame'],
            'smplx_root_pose': [],
            'smplx_body_pose': [],
            'smplx_lhand_pose': [],
            'smplx_rhand_pose': [],
            'smplx_jaw_pose': [],
            'smplx_shape': frame['smplx_shape'],
            'smplx_expr': frame['smplx_expr'],
            'cam_trans': frame['cam_trans']
        }
        
        # 处理需要转换的参数
        pose_keys = ['smplx_root_pose', 'smplx_body_pose', 'smplx_lhand_pose', 'smplx_rhand_pose', 'smplx_jaw_pose']
        
        for key in pose_keys:
            euler_pose = frame[key]
            axis_angle_pose = []
            
            for euler in euler_pose:
                axis_angle = euler_to_axis_angle(np.array(euler))
                axis_angle_pose.extend(axis_angle.tolist())
            
            axis_angle_frame[key] = axis_angle_pose
        
        smoothed_params.append(axis_angle_frame)
    
    return smoothed_params

# --- 6. 主流程：加载数据、渲染视频 ---
if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SMPLest_X Render')
    parser.add_argument('--input_paths', type=str, default='video_clips', help='输入视频路径或通配符模式')
    parser.add_argument('--output_paths', type=str, default='output_videos', help='输出视频路径')
    parser.add_argument('--smooth', type=bool, default=True, help='是否平滑动画数据')
    args = parser.parse_args()
    
    smooth = args.smooth
    renderer, scene = init_renderer()
    input_paths = glob.glob(f"{args.input_paths}/*.json")

    idle_pose_path = r"idle_pose\idle_pose.json"
    with open(idle_pose_path, 'r') as f:
        idle_pose_smplx_params_list = json.load(f)
    num_frames = len(idle_pose_smplx_params_list)
    # idle pose 有三百帧
    idle_animation_data = process_json_animation_file(idle_pose_smplx_params_list, num_frames, eular=True)

    for input_path in input_paths:
        basename = os.path.basename(input_path)
        output_video_name = basename.replace(".json", ".mp4")
        output_paths = os.path.join(args.output_paths, output_video_name)
        os.makedirs(os.path.dirname(output_paths), exist_ok=True)
        
        with open(input_path, 'r') as f:
            smplx_params_list = json.load(f)
        
        if smooth:
            new_animation_data = process_json_animation_file(smplx_params_list, num_frames, eular=True, upper_body=True)
            new_animation_data = smooth_animation_data(new_animation_data, window_size=5)
            new_animation_data = blend_animations(new_animation_data, idle_animation_data, num_frames)
        else:
            new_animation_data = process_json_animation_file(smplx_params_list, num_frames, eular=True, upper_body=False)

        frame_num = len(new_animation_data)        # 初始化视频写入器 (使用 mp4v 编码)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_paths, fourcc, 25, (RENDER_RESOLUTION, RENDER_RESOLUTION))

        print("开始渲染视频...")
        for i in tqdm.tqdm(range(frame_num)):
            # 渲染当前帧 (得到的是 RGB 图像)
            rgb_image = render_frame(new_animation_data[i], renderer, scene)
            video_writer.write(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        video_writer.release()

    renderer.delete()