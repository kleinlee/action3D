import cv2
import numpy as np
from utils._rotation_utils import EulurAngle

smplx_parents_index_list = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    15,
    15,
    15,
    20,
    25,
    26,
    20,
    28,
    29,
    20,
    31,
    32,
    20,
    34,
    35,
    20,
    37,
    38,
    21,
    40,
    41,
    21,
    43,
    44,
    21,
    46,
    47,
    21,
    49,
    50,
    21,
    52,
    53
]

lower_body_index = [0, 1, 2, 4, 5, 7, 8, 10, 11]

# 2. SMPLX 55关节的标准名称 (按索引严格对应)
smplx_joint_names = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eye',
    'right_eye',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3'
]



# --- 核心：SMPLX 公式逻辑 ---
pack = lambda x: np.hstack([np.zeros((4, 3)), x.reshape((4, 1))])
with_zeros = lambda x: np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

def get_verts(model_dict, pose, betas):
    v_template = model_dict['v_template']
    shapedirs = model_dict['shapedirs']
    posedirs = model_dict['posedirs']
    J_regressor = model_dict['J_regressor']
    weights = model_dict['weights']
    kintree_table = model_dict['kintree_table']
    
    # 1. Add shape blend shapes
    v_shaped = v_template + shapedirs[:,:,:10].dot(betas)

    # 2. Infer joint locations
    J_tmpx = J_regressor.dot(v_shaped[:, 0])
    J_tmpy = J_regressor.dot(v_shaped[:, 1])
    J_tmpz = J_regressor.dot(v_shaped[:, 2])
    J = np.vstack((J_tmpx, J_tmpy, J_tmpz)).T

    # 3. Add pose blend shapes
    pose = pose.reshape((-1, 3))
    pose_rotmats = []
    for i in range(pose.shape[0]):
        pose_rotmats.append(cv2.Rodrigues(pose[i])[0])
    
    tmp = np.hstack([(pose_rotmats[i] - np.eye(3)).flatten() for i in range(1, len(pose_rotmats))]).flatten()
    v_posed = v_shaped + posedirs.dot(tmp)

    # 4. Forward Kinematics
    results = {}
    parent = {kintree_table[1, i]: kintree_table[0, i] for i in range(1, kintree_table.shape[1])}

    results[0] = with_zeros(np.hstack((pose_rotmats[0], J[0, :].reshape((3, 1)))))
    
    for i in range(1, kintree_table.shape[1]):
        results[i] = results[parent[i]].dot(with_zeros(np.hstack((pose_rotmats[i],
                                                                  ((J[i, :] - J[parent[i], :]).reshape((3, 1)))))))

    results = [results[i] for i in sorted(results.keys())]
    Jtr = np.vstack(np.array([g[:3, 3] for g in results]))
    
    results2 = [results[i] - (pack(results[i].dot(np.hstack(((J[i, :]), 0))))) for i in range(len(results))]
    results = results2
    
    # 5. Skinning
    result = np.dstack(results)
    T = result.dot(weights.T)
    
    v = v_posed
    rest_shape_h = np.vstack((v.T, np.ones((1, v.shape[0]))))
    
    v = (T[:, 0, :] * rest_shape_h[0, :].reshape((1, -1)) +
         T[:, 1, :] * rest_shape_h[1, :].reshape((1, -1)) +
         T[:, 2, :] * rest_shape_h[2, :].reshape((1, -1)) +
         T[:, 3, :] * rest_shape_h[3, :].reshape((1, -1))).T
         
    return v[:, :3], Jtr


# 处理 JSON 动画文件数据
def process_json_animation_file(smplx_params_list, num_frames, eular=None, upper_body=False):
    """处理 JSON 动画文件数据
    smplx_params_list: 动画参数列表
    num_frames: 处理的帧数
    upper_body: 是否仅处理上半身，False 表示处理下半身
    eular: 控制是否转换为欧拉角，None 表示不转换，True 表示转换
    """
    # 截取前num_frames帧
    animation_subset = smplx_params_list[:num_frames]
    
    # 转换为与新动画相同的格式
    animation_data = []
    
    if eular:
        # 转换为欧拉角并处理角度连续性
        num_joints = 55
        previous_euler = [np.array([0.0, 0.0, 0.0]) for _ in range(num_joints)]
        
        for frame_index, feature_dict in enumerate(animation_subset):
            # 提取并拼接完整的 55 关节轴角 (1x165)
            global_orient = np.array(feature_dict['smplx_root_pose'], dtype=np.float32).reshape(1, 3)
            body_pose = np.array(feature_dict['smplx_body_pose'], dtype=np.float32).reshape(21, 3)
            if upper_body:
                body_pose[lower_body_index] = 0
            jaw_pose = np.array(feature_dict['smplx_jaw_pose'], dtype=np.float32).reshape(1, 3) if 'smplx_jaw_pose' in feature_dict else np.zeros([1, 3], dtype=np.float32)
            eye_pose = np.zeros([2, 3], dtype=np.float32)
            left_hand_pose = np.array(feature_dict['smplx_lhand_pose'], dtype=np.float32).reshape(15, 3) if 'smplx_lhand_pose' in feature_dict else np.zeros([15, 3], dtype=np.float32)
            right_hand_pose = np.array(feature_dict['smplx_rhand_pose'], dtype=np.float32).reshape(15, 3) if 'smplx_rhand_pose' in feature_dict else np.zeros([15, 3], dtype=np.float32)
        
            full_pose = np.concatenate([global_orient, body_pose, jaw_pose, eye_pose, left_hand_pose, right_hand_pose], axis=0) # 55*3
            
            # 转换为欧拉角并处理角度连续性
            euler_pose = []
            for j_idx in range(num_joints):
                bone_eular = full_pose[j_idx]
                # 过滤掉全0的微小误差
                if np.linalg.norm(bone_eular) < 1e-6:
                    euler = np.array([0.0, 0.0, 0.0])
                else:
                    rot_mat, _ = cv2.Rodrigues(bone_eular)
                    euler = EulurAngle(rot_mat)
                
                if frame_index == 0:
                    for i in range(3):
                        diff = euler[i]
                        if euler[i] < -90:
                            euler[i] += 360
                else:
                    # 处理角度连续性问题，确保相邻帧之间的角度变化最小
                    prev_euler = previous_euler[j_idx]
                    for i in range(3):
                        diff = euler[i] - prev_euler[i]
                        if diff > 180:
                            euler[i] -= 360
                        elif diff < -180:
                            euler[i] += 360
                previous_euler[j_idx] = euler
                euler_pose.append(euler)
            
            animation_data.append(euler_pose)
    else:
        # 保持轴角格式
        for feature_dict in animation_subset:
            # 提取并拼接完整的 55 关节轴角 (1x165)
            global_orient = np.array(feature_dict['smplx_root_pose'], dtype=np.float32).reshape(1, 3)
            body_pose = np.array(feature_dict['smplx_body_pose'], dtype=np.float32).reshape(21, 3)
            jaw_pose = np.array(feature_dict['smplx_jaw_pose'], dtype=np.float32).reshape(1, 3) if 'smplx_jaw_pose' in feature_dict else np.zeros([1, 3], dtype=np.float32)
            eye_pose = np.zeros([2, 3], dtype=np.float32)
            left_hand_pose = np.array(feature_dict['smplx_lhand_pose'], dtype=np.float32).reshape(15, 3) if 'smplx_lhand_pose' in feature_dict else np.zeros([15, 3], dtype=np.float32)
            right_hand_pose = np.array(feature_dict['smplx_rhand_pose'], dtype=np.float32).reshape(15, 3) if 'smplx_rhand_pose' in feature_dict else np.zeros([15, 3], dtype=np.float32)
        
            full_pose = np.concatenate([global_orient, body_pose, jaw_pose, eye_pose, left_hand_pose, right_hand_pose], axis=0) # 55*3
            animation_data.append(full_pose)
    
    animation_data = np.array(animation_data)
    return animation_data

# 添加平滑函数
def smooth_data(data_list, window_size=5):
    """对数据进行加权滑动窗口平滑"""
    if len(data_list) <= window_size:
        return data_list
    
    # 定义5帧平滑的权重
    weights = [0.1, 0.2, 0.4, 0.2, 0.1]
    
    smoothed_data = []
    for i in range(len(data_list)):
        # 计算窗口的起始和结束位置
        start = max(0, i - window_size // 2)
        end = min(len(data_list), i + window_size // 2 + 1)
        window = data_list[start:end]
        window_size_actual = len(window)
        
        # 根据实际窗口大小调整权重
        if window_size_actual < window_size:
            # 当窗口大小小于5时，使用均匀权重
            adjusted_weights = [1.0 / window_size_actual] * window_size_actual
        else:
            adjusted_weights = weights
        
        # 对每个字段进行平滑
        smoothed_frame = {}
        for key in window[0].keys():
            # 收集窗口内所有帧的该字段值
            values = [frame[key] for frame in window]
            
            # 对列表类型的字段进行平滑
            if isinstance(values[0], list):
                # 转置列表，对每个元素单独平滑
                transposed = list(zip(*values))
                smoothed_transposed = []
                for elem in transposed:
                    weighted_sum = sum(v * w for v, w in zip(elem, adjusted_weights))
                    smoothed_transposed.append(weighted_sum)
                smoothed_frame[key] = smoothed_transposed
            else:
                # 对单个值进行平滑
                weighted_sum = sum(v * w for v, w in zip(values, adjusted_weights))
                smoothed_frame[key] = weighted_sum
        
        smoothed_data.append(smoothed_frame)
    
    return smoothed_data

# 平滑处理动画数据
def smooth_animation_data(animation_data, window_size=5):
    """对动画数据进行加权平滑处理
    animation_data: 动画数据列表，每个元素是一个帧的动画数据（欧拉角格式）
    window_size: 平滑窗口大小
    """
    # 定义5帧平滑的权重
    weights = [0.1, 0.2, 0.4, 0.2, 0.1]
    
    num_joints = len(animation_data[0])
    num_frames = len(animation_data)
    
    # 对每个骨骼的旋转数据进行平滑
    smoothed_data = []
    for i in range(num_frames):
        smoothed_frame = []
        for j in range(num_joints):
            # 计算窗口的起始和结束位置
            start = max(0, i - window_size // 2)
            end = min(num_frames, i + window_size // 2 + 1)
            window = animation_data[start:end]
            window_size_actual = len(window)
            
            if window_size_actual < window_size:
                adjusted_weights = [1.0 / window_size_actual] * window_size_actual
            else:
                adjusted_weights = weights
            
            # 对旋转的每个分量进行加权平均
            smoothed_rotate_comp = [0.0, 0.0, 0.0]
            for k in range(3):
                weighted_sum = sum(w * window[t][j][k] for t, w in enumerate(adjusted_weights))
                smoothed_rotate_comp[k] = weighted_sum
            smoothed_frame.append(smoothed_rotate_comp)
        smoothed_data.append(smoothed_frame)
    
    return smoothed_data



# 实现渐入渐出融合
def blend_animations(animation_data, idle_animation_data, num_frames):
    """实现动画的渐入渐出融合
    前15帧：从idle pose渐入到新动画
    后15帧：从新动画渐出到idle pose
    """
    blend_frames = 12
    
    num_joints = len(animation_data[0])
    num_frames_total = len(animation_data)
    
    # 对每个帧和骨骼进行融合
    blended_data = []
    for i in range(num_frames_total):
        blended_frame = []
        for j in range(num_joints):
            if i < blend_frames:
                # 渐入阶段：从idle pose到新动画
                alpha = i / (blend_frames - 1) if blend_frames > 1 else 1.0
                # 获取idle pose对应帧的旋转（已转换为欧拉角）
                idle_euler = idle_animation_data[i][j]
                # 线性插值
                blended = [
                    (1 - alpha) * idle_euler[0] + alpha * animation_data[i][j][0],
                    (1 - alpha) * idle_euler[1] + alpha * animation_data[i][j][1],
                    (1 - alpha) * idle_euler[2] + alpha * animation_data[i][j][2]
                ]
            elif i >= num_frames_total - blend_frames:
                # 渐出阶段：从新动画到idle pose
                alpha = (num_frames_total - i - 1) / (blend_frames - 1) if blend_frames > 1 else 1.0
                # 获取idle pose对应帧的旋转（使用前num_frames帧中的对应位置）
                idle_idx = min(i - (num_frames_total - num_frames), num_frames - 1)
                idle_euler = idle_animation_data[idle_idx][j]
                # 线性插值
                blended = [
                    alpha * animation_data[i][j][0] + (1 - alpha) * idle_euler[0],
                    alpha * animation_data[i][j][1] + (1 - alpha) * idle_euler[1],
                    alpha * animation_data[i][j][2] + (1 - alpha) * idle_euler[2]
                ]
            else:
                # 中间阶段：使用原始动画
                blended = animation_data[i][j]
            
            blended_frame.append(blended)
        blended_data.append(blended_frame)
    blended_data = np.array(blended_data)
    
    return blended_data

def prepare_weights_data(weights, max_weights_per_vertex=4):
    """
    将 weights (Num_verts, Num_bones) 转换为 FBX 需要的 weights_data 字典
    请在外部先将其转置为 (10475, 55) 再传入此函数。
    """
    # 假设传入的 W 已经是 (Num_verts, Num_bones)
    W = weights
    
    weights_data = {}
    num_bones = W.shape[1]
    
    # 对每个顶点，只保留权重最大的前 N 个骨骼（FBX通常限制每个顶点最多受4-8个骨骼影响）
    for bone_idx in range(num_bones):
        weights_data[bone_idx] = ([], [])
        
    for vert_idx in range(W.shape[0]):
        row = W[vert_idx]
        # 获取权重最大的几个骨骼索引
        top_bones = np.argsort(row)[-max_weights_per_vertex:]
        
        selected_weights = row[top_bones]
        # 归一化（避免除以零）
        total = np.sum(selected_weights)
        if total > 0:
            selected_weights /= total
        # 更新 weights_data
        for bone_idx, w in zip(top_bones, selected_weights):
            if w > 1e-5:
                weights_data[bone_idx][0].append(vert_idx)
                weights_data[bone_idx][1].append(w)
                
    # 转换为 numpy 数组
    for bone_idx in weights_data:
        idxs, ws = weights_data[bone_idx]
        weights_data[bone_idx] = (np.array(idxs, dtype=np.int32), np.array(ws, dtype=np.float64))
        
    return weights_data