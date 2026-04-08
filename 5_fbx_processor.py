import re
import numpy as np
from render import init_renderer,render_frame
import cv2
import tqdm
import argparse
from utils.smplx_utils import smplx_joint_names

def extract_fbx_keyframes(fbx_file_path):
    """
    读取ASCII格式的FBX文件，提取所有AnimationCurve的关键帧时间和数值
    
    参数:
        fbx_file_path: FBX文件路径
    
    返回:
        dict: 包含每个AnimationCurve的关键帧数据的字典，每个元素为字典
            {
                'key_times': list,         # 关键帧时间列表
                'key_values': list,        # 关键帧数值列表
                'key_count': int           # 关键帧数量
            }
    """
    # 打开并读取FBX文件
    with open(fbx_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 初始化结果列表
    animation_curves = {}
    
    # 匹配AnimationCurve ID
    animation_curve_ids = re.findall(r'AnimationCurve: ([^,]+), "AnimCurve::", ""', content)
    
    # 匹配KeyTime块
    key_time_pattern = re.compile(r'KeyTime:\s*\*([0-9]+)\s*\{\s*a:\s*([\s\S]*?)\s*\}', re.MULTILINE)
    key_time_matches = key_time_pattern.findall(content)
    
    # 匹配KeyValueFloat块
    key_value_pattern = re.compile(r'KeyValueFloat:\s*\*([0-9]+)\s*\{\s*a:\s*([\s\S]*?)\s*\}', re.MULTILINE)
    key_value_matches = key_value_pattern.findall(content)
    
    # 确保三个列表长度相同
    min_length = min(len(animation_curve_ids), len(key_time_matches), len(key_value_matches))
    
    # 处理每个AnimationCurve
    for i in range(min_length):
        curve_id = animation_curve_ids[i].strip()
        key_count, key_times_str = key_time_matches[i]
        key_count = int(key_count)
        _, key_values_str = key_value_matches[i]
        
        # 解析关键帧时间（处理多行和空格）
        key_times_str = re.sub(r'\s+', '', key_times_str)
        key_times = [float(t) for t in key_times_str.split(',') if t]
        
        # 解析关键帧数值（处理多行和空格）
        key_values_str = re.sub(r'\s+', '', key_values_str)
        key_values = [float(v) for v in key_values_str.split(',') if v]
        
        # 验证数据一致性
        if len(key_times) == key_count and len(key_values) == key_count:
            animation_curves[curve_id] = {
                'key_times': key_times,
                'key_values': key_values,
                'key_count': key_count
            }
        else:
            print(f"警告: AnimationCurve {curve_id} 的关键帧数量不一致")
    
    return animation_curves


def sort_animation_curves_by_smplx_order(animation_curves, fbx_file_path):
    """
    根据SMPLX骨骼顺序对AnimationCurve进行排序
    
    参数:
        animation_curves: 提取到的AnimationCurve字典
        fbx_file_path: FBX文件路径
    
    返回:
        dict: 按SMPLX骨骼顺序排序后的AnimationCurve字典
    """
    
    # 打开并读取FBX文件
    with open(fbx_file_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    
    # 提取AnimCurveNode和Model的对应关系
    # 创建AnimCurveNode到骨骼名称的映射
    bone_to_anim_curve_node = {}
    AnimCurveNode_index = None
    rows = 0
    while rows < len(content):
        i = content[rows]
        if i.find(";AnimCurveNode::R, Model") > -1:
            AnimCurveNode_index = content[rows+1].split(",")[1]
            bone_name = i.split("Model::CC_Base_")[-1][:-1]
            bone_to_anim_curve_node[bone_name] = AnimCurveNode_index
            rows = rows + 2
        else:
            rows = rows + 1
    print(bone_to_anim_curve_node)

    # 创建AnimCurveNode id到 curve id的映射
    anim_curve_node_to_curve = {}
    rows = 0
    while rows < len(content):
        i = content[rows]
        if i.find(";AnimCurve::, AnimCurveNode::R") > -1:
            AnimCurve_index = content[rows+1].split(",")[1]
            AnimCurveNode_index = content[rows+1].split(",")[2]
            axis = content[rows+1].split("d|")[-1][0]
            anim_curve_node_to_curve[AnimCurveNode_index + axis] = AnimCurve_index
            rows = rows + 2
        else:
            rows = rows + 1


    # 创建骨骼名称到索引的映射
    bone_to_index = {bone: i for i, bone in enumerate(smplx_joint_names)}
    
    new_animation_curves_list = []
    for bone_name in smplx_joint_names:
        anim_curve_node = bone_to_anim_curve_node[bone_name]
        curve = animation_curves[anim_curve_node_to_curve[anim_curve_node + "X"]]
        curve['bone_name'] = bone_name
        curve['axis'] = 'X'
        curve_id = anim_curve_node_to_curve[anim_curve_node + "X"]
        new_animation_curves_list.append(curve)
        curve = animation_curves[anim_curve_node_to_curve[anim_curve_node + "Y"]]
        curve['bone_name'] = bone_name
        curve['axis'] = 'Y'
        new_animation_curves_list.append(curve)
        curve = animation_curves[anim_curve_node_to_curve[anim_curve_node + "Z"]]
        curve['bone_name'] = bone_name
        curve['axis'] = 'Z'
        new_animation_curves_list.append(curve)
    
    return new_animation_curves_list


def linear_interpolation(x, x0, y0, x1, y1):
    """
    线性插值函数
    
    参数:
        x: 要插值的x值
        x0, y0: 第一个点的坐标
        x1, y1: 第二个点的坐标
    
    返回:
        插值得到的y值
    """
    if x0 == x1:
        return y0
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))


def interpolate_animation_curve(key_times, key_values, frame_time, num_frames):
    """
    对单个AnimationCurve进行线性插值，得到每一帧的值
    
    参数:
        key_times: 关键帧时间列表
        key_values: 关键帧数值列表
        frame_time: 每帧的时间间隔
        num_frames: 总帧数
    
    返回:
        插值后的帧数值列表
    """
    interpolated_values = []
    
    for frame in range(num_frames):
        current_time = frame * frame_time
        
        # 找到current_time所在的关键帧区间
        if current_time <= key_times[0]:
            # 当前时间早于第一个关键帧
            interpolated_values.append(key_values[0])
        elif current_time >= key_times[-1]:
            # 当前时间晚于最后一个关键帧
            interpolated_values.append(key_values[-1])
        else:
            # 找到current_time所在的区间
            for i in range(len(key_times) - 1):
                if key_times[i] <= current_time <= key_times[i+1]:
                    # 线性插值
                    value = linear_interpolation(
                        current_time, 
                        key_times[i], key_values[i], 
                        key_times[i+1], key_values[i+1]
                    )
                    interpolated_values.append(value)
                    break
    
    return interpolated_values


def process_fbx_file(fbx_file_path, frame_time=1539538600):
    """
    处理ASCII格式的FBX文件，提取关键帧并进行线性插值
    
    参数:
        fbx_file_path: FBX文件路径
        frame_time: 每帧的时间间隔，默认值为1539538600
    
    返回:
        numpy.ndarray: 插值后的数组，形状为[frame_num, 165]
    """
    # 提取关键帧数据
    print(f"正在提取FBX文件中的关键帧数据...")
    animation_curves = extract_fbx_keyframes(fbx_file_path)
    
    print(f"共提取到 {len(animation_curves)} 个AnimationCurve")
    
    # 按SMPLX骨骼顺序排序
    print("正在按SMPLX骨骼顺序排序AnimationCurve...")
    new_animation_curves_list = sort_animation_curves_by_smplx_order(animation_curves, fbx_file_path)
    
    # 计算总帧数
    # 找到最大的关键帧时间
    max_key_time = 0
    for curve in new_animation_curves_list:
        if curve['key_times'][-1] > max_key_time:
            max_key_time = curve['key_times'][-1]
    
    # 计算总帧数
    num_frames = int(max_key_time / frame_time) + 1
    
    print(f"总帧数: {num_frames}")
    
    # 对每个AnimationCurve进行插值
    interpolated_data = []
    for i, curve in enumerate(new_animation_curves_list):
        print(f"处理AnimationCurve {i+1}/{len(new_animation_curves_list)}")
        key_times = curve['key_times']
        key_values = curve['key_values']
        
        # 插值得到每一帧的值
        frame_values = interpolate_animation_curve(key_times, key_values, frame_time, num_frames)
        interpolated_data.append(frame_values)
    
    # 转换为numpy数组，形状为[frame_num, 165]
    interpolated_array = np.array(interpolated_data).T
    
    print(f"插值后数组形状: {interpolated_array.shape}")
    
    # 打印前几帧的数据
    print("\n前3帧的数据:")
    for i in range(min(3, num_frames)):
        print(f"Frame {i}: {interpolated_array[i][:5]}...")
    
    return interpolated_array


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='FBX Processor')
    parser.add_argument('input_path', type=str, help='FBX文件路径')
    args = parser.parse_args()
    
    # 使用命令行输入的路径
    fbx_file_path = args.input_path
    interpolated_array = process_fbx_file(fbx_file_path)
    np.save(fbx_file_path.replace('.fbx', '.npy'), interpolated_array)
    print(f"结果已保存到 {fbx_file_path.replace('.fbx', '.npy')}")

    interpolated_array = interpolated_array.reshape(-1, 55, 3)
    frame_num = interpolated_array.shape[0]


    renderer, scene = init_renderer()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    OUTPUT_VIDEO_PATH = fbx_file_path.replace('.fbx', '.mp4')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 25, (512, 512))

    print("开始渲染视频...")
    for i in tqdm.tqdm(range(frame_num)):
        # 渲染当前帧 (得到的是 RGB 图像)
        rgb_image = render_frame(interpolated_array[i], renderer, scene)
        video_writer.write(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    video_writer.release()
    renderer.delete()

if __name__ == "__main__":
    main()