import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation as R
import json
import os
import cv2
import tqdm
from render import init_renderer, RENDER_RESOLUTION, render_frame
from utils.smplx_utils import process_json_animation_file

SMPLX_MODEL_PATH = r"SMPLest_X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz"
OUTPUT_OBJ_PATH  = "idle_pose/idle_pose.obj"
OUTPUT_JSON_PATH = "idle_pose/idle_pose.json"
OUTPUT_VIDEO_PATH = OUTPUT_JSON_PATH.replace(".json", ".mp4")

NUM_FRAMES = 300  # 300帧

# ========== 创建模型 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smplx_model = smplx.create(
    SMPLX_MODEL_PATH,
    model_type='smplx',
    gender='neutral',
    use_face_contour=False,
    num_betas=10,
    use_pca=False,      
    flat_hand_mean=True,
).to(device)

# ========== 工具函数 ==========
def euler_deg_to_axis_angle(euler_deg, order='xyz'):
    """欧拉角（度）→ 轴角（弧度），SMPLX使用的旋转表示"""
    rot = R.from_euler(order, euler_deg, degrees=True)
    return rot.as_rotvec().astype(np.float32)

def write_obj(filepath, vertices, faces):
    """将顶点和面片写入OBJ文件"""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"[✓] OBJ 已保存: {os.path.abspath(filepath)}")

# ========== 构建姿态参数（T-pose基准，全零） ==========
global_orient     = np.zeros((1, 3),  dtype=np.float32)
body_pose         = np.zeros((1, 63), dtype=np.float32)
left_hand_pose    = np.zeros((1, 45), dtype=np.float32)
right_hand_pose   = np.zeros((1, 45), dtype=np.float32)

global_orient[0] = np.array([-np.pi, 0, 0])

# --- 大臂 ---
body_pose[0, 45:48] = euler_deg_to_axis_angle([25, -10, -55])   # 左侧大臂
body_pose[0, 48:51] = euler_deg_to_axis_angle([25,  10,  55])   # 右侧大臂

# --- 小臂 ---
body_pose[0, 51:54] = euler_deg_to_axis_angle([75, -65, -80])   # 左侧小臂
body_pose[0, 54:57] = euler_deg_to_axis_angle([75,  65,  90])   # 右侧小臂

# --- 手腕 ---
body_pose[0, 57:60] = euler_deg_to_axis_angle([-10, -25, -15])  # 左侧手腕
body_pose[0, 60:63] = euler_deg_to_axis_angle([-10,  25,   5])  # 右侧手腕

# --- 左手 ---
left_hand_pose[0,  0: 3] = euler_deg_to_axis_angle([0,  10, -20])   # 食指 1
left_hand_pose[0,  3: 6] = euler_deg_to_axis_angle([0,   0, -10])   # 食指 2
left_hand_pose[0,  9:12] = euler_deg_to_axis_angle([0,   0, -20])   # 中指 1
left_hand_pose[0, 12:15] = euler_deg_to_axis_angle([0,   0, -10])   # 中指 2
left_hand_pose[0, 18:21] = euler_deg_to_axis_angle([0, -25, -20])   # 小拇指 1
left_hand_pose[0, 21:24] = euler_deg_to_axis_angle([0,   0, -10])   # 小拇指 2
left_hand_pose[0, 27:30] = euler_deg_to_axis_angle([0, -10, -20])   # 无名指 1
left_hand_pose[0, 30:33] = euler_deg_to_axis_angle([0,   0, -10])   # 无名指 2
left_hand_pose[0, 36:39] = euler_deg_to_axis_angle([0,  30,   0])   # 大拇指 1

# --- 右手 ---
right_hand_pose[0,  0: 3] = euler_deg_to_axis_angle([0, -10,  15])   # 食指 1
right_hand_pose[0,  9:12] = euler_deg_to_axis_angle([0,   0,  15])   # 中指 1
right_hand_pose[0, 18:21] = euler_deg_to_axis_angle([0,  25,  15])   # 小拇指 1
right_hand_pose[0, 27:30] = euler_deg_to_axis_angle([0,  10,  15])   # 无名指 1
right_hand_pose[0, 36:39] = euler_deg_to_axis_angle([0, -30,   0])   # 大拇指 1

# ========== 送入 SMPLX 生成网格 ==========
with torch.no_grad():
    body_output = smplx_model(
        global_orient  = torch.tensor(global_orient,   device=device),
        body_pose      = torch.tensor(body_pose,       device=device),
        left_hand_pose = torch.tensor(left_hand_pose,  device=device),
        right_hand_pose= torch.tensor(right_hand_pose, device=device),
        betas          = torch.zeros(1, 10, dtype=torch.float32, device=device),
    )

vertices = body_output.vertices.detach().cpu().numpy().squeeze()
faces    = smplx_model.faces
write_obj(OUTPUT_OBJ_PATH, vertices, faces)

# ========== 提取基础参数用于动画 ==========
smplx_root_pose = global_orient[0]
smplx_jaw_pose  = np.zeros(3, dtype=np.float32)
smplx_shape     = np.zeros(10, dtype=np.float32)
smplx_expr      = np.zeros(10, dtype=np.float32)
cam_trans       = np.zeros(3, dtype=np.float32)

base_body_pose = body_pose[0].copy()
base_left_hand_pose = left_hand_pose[0].copy()
base_right_hand_pose = right_hand_pose[0].copy()

# ========== 动画生成逻辑 ==========
results = []

for frame_index in range(NUM_FRAMES):
    t = frame_index / NUM_FRAMES
    phase = 2 * np.pi * t
    
    current_body_pose = base_body_pose.copy()
    
    # -------------------------------------------------------------
    # 1. 呼吸感 (影响 Spine 和 Collar)
    #    表现：胸腔前后起伏 + 微弱的侧向膨胀 + 肩膀随呼吸上下浮动
    # -------------------------------------------------------------
    breath_phase = phase * 2.0  # 平稳呼吸频率
    breath_x = 0.012 * np.sin(breath_phase) 
    breath_y = 0.005 * np.sin(breath_phase + 0.5) # 侧向微小膨胀
    
    # Spines
    current_body_pose[6:9][0] += breath_x               # spine1 X
    current_body_pose[6:9][1] += breath_y               # spine1 Y (侧向)
    current_body_pose[15:18][0] += breath_x * 0.4       # spine2 X
    # current_body_pose[24:27][0] += breath_x * 0.4       # spine3 X
    
    # Collars (让手臂根部有附着感，而不是飘在空中)
    current_body_pose[36:39][0] += breath_x * 0.3       # left_collar X
    current_body_pose[39:42][0] += breath_x * 0.3       # right_collar X

    # -------------------------------------------------------------
    # 2. 真实的重心转移 (影响 Spine微倾 + 膝盖微屈)
    #    原理：绝对不旋转大腿骨，而是让脊柱底部极微弱侧倾，
    #         并让左右膝盖产生微小的交替弯曲来"吃掉"重心偏移。
    # -------------------------------------------------------------
    sway_phase = phase * 0.8  # 极慢的重心转移周期
    sway_z = 0.005 * np.sin(sway_phase) 
    
    # 脊柱随重心极微弱侧倾
    current_body_pose[6:9][2] += sway_z
    current_body_pose[15:18][2] += sway_z * 0.5
    
    # 膝盖交替微屈 (idx 3: left_knee -> 9:12, idx 4: right_knee -> 12:15)
    knee_sway = 0.006 * np.sin(sway_phase)
    current_body_pose[9:12][0] += knee_sway    # 左膝微曲
    current_body_pose[12:15][0] -= knee_sway   # 右膝反向微曲

    # -------------------------------------------------------------
    # 3. 更丰富的头部微动 (三维：点头 + 环视 + 歪头)
    # -------------------------------------------------------------
    head_x = np.pi/18 + 0.008 * np.sin(phase * 1.2 + 0.5)  # 前后点头 (X轴)
    head_y = 0.02 * np.sin(phase * 0.6 + 1.0)  # 左右缓慢环视 (Y轴)
    head_z = 0.008 * np.sin(phase * 0.9 + 2.0)  # 微微歪头 (Z轴)
    
    # Neck 补偿
    current_body_pose[33:36][0] -= head_x * 0.3
    current_body_pose[33:36][1] -= head_y * 0.3
    current_body_pose[33:36][2] -= head_z * 0.3
    
    # Head 主动作
    current_body_pose[42:45][0] += head_x
    current_body_pose[42:45][1] += head_y
    current_body_pose[42:45][2] += head_z

    # -------------------------------------------------------------
    # 组装当前帧数据
    # -------------------------------------------------------------
    frame_result = {
        'frame': frame_index,
        'smplx_root_pose': smplx_root_pose.tolist(),
        'smplx_body_pose': current_body_pose.tolist(),
        'smplx_lhand_pose': base_left_hand_pose.tolist(),
        'smplx_rhand_pose': base_right_hand_pose.tolist(),
        'smplx_jaw_pose': smplx_jaw_pose.tolist(),
        'smplx_shape': smplx_shape.tolist(),
        'smplx_expr': smplx_expr.tolist(),
        'cam_trans': cam_trans.tolist()
    }
    results.append(frame_result)

# ========== 导出 JSON ==========
with open(OUTPUT_JSON_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"[✓] Idle 动画 JSON 已成功保存: {OUTPUT_JSON_PATH} (共 {NUM_FRAMES} 帧)")

renderer, scene = init_renderer()
frame_num = len(results)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 25.0, (RENDER_RESOLUTION, RENDER_RESOLUTION))
with open(OUTPUT_JSON_PATH, 'r') as f:
    smplx_params_list = json.load(f)
new_animation_data = process_json_animation_file(smplx_params_list, frame_num, eular=True, upper_body=False)
print("开始渲染视频...")
for i in tqdm.tqdm(range(frame_num)):
    rgb_image = render_frame(new_animation_data[i], renderer, scene)
    video_writer.write(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        
video_writer.release()
renderer.delete()
print(f"[✓] 视频渲染完成: {OUTPUT_VIDEO_PATH}")
