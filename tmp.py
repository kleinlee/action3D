import os
import subprocess
import re

def get_video_duration(filepath):
    """获取视频时长（秒）"""
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{filepath}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return float(result.stdout.strip())

def main():
    video_dir = r"E:\Code\3Dpose\FBX\video_clips"
    output_path = r"E:\Code\3Dpose\FBX\output_merged.mp4"
    
    # 获取所有mp4文件
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    videos.sort()  # 排序确保顺序一致
    video_paths = [os.path.join(video_dir, v) for v in videos]
    
    if len(video_paths) != 18:
        print(f"警告：找到 {len(video_paths)} 个视频，期望 18 个")
    
    # 找到最小时长
    durations = []
    for vp in video_paths:
        try:
            dur = get_video_duration(vp)
            durations.append(dur)
            print(f"{os.path.basename(vp)}: {dur:.2f}秒")
        except Exception as e:
            print(f"读取 {vp} 失败: {e}")
            return
    
    min_duration = min(durations)
    print(f"\n最小时长: {min_duration:.2f}秒")
    
    # 创建临时目录存放处理后的视频
    temp_dir = os.path.join(video_dir, "temp_processed")
    os.makedirs(temp_dir, exist_ok=True)
    
    processed_videos = []
    
    # 处理每个视频：截取最小时长 + 缩放到320x320
    for i, vp in enumerate(video_paths):
        temp_output = os.path.join(temp_dir, f"processed_{i:02d}.mp4")
        processed_videos.append(temp_output)
        
        # ffmpeg命令：截取前min_duration秒，缩放到320x320，保持宽高比并填充黑边到320x320
        cmd = (f'ffmpeg -i "{vp}" -t {min_duration} '
               f'-vf "scale=320:320:force_original_aspect_ratio=1,pad=320:320:(ow-iw)/2:(oh-ih)/2" '
               f'-c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k '
               f'-y "{temp_output}"')
        
        print(f"处理视频 {i+1}/18: {os.path.basename(vp)}")
        os.system(cmd)
    
    # 方法1：使用hstack和vstack组合（更兼容）
    print("\n拼接视频中...")
    
    # 先拼接每行的6个视频
    row_videos = []
    for row in range(3):
        start_idx = row * 6
        # 构建该行的hstack命令
        row_inputs = "".join([f'-i "{processed_videos[start_idx + col]}" ' for col in range(6)])
        row_filter = f'hstack=inputs=6'
        row_output = os.path.join(temp_dir, f"row_{row}.mp4")
        
        cmd_row = f'ffmpeg {row_inputs} -filter_complex "{row_filter}" -c:v libx264 -preset fast -crf 23 -y "{row_output}"'
        print(f"拼接第{row+1}行...")
        os.system(cmd_row)
        row_videos.append(row_output)
    
    # 再垂直拼接3行
    vstack_inputs = "".join([f'-i "{row_video}" ' for row_video in row_videos])
    vstack_filter = f'vstack=inputs=3'
    temp_merged = os.path.join(temp_dir, "temp_merged.mp4")
    
    cmd_merge = f'ffmpeg {vstack_inputs} -filter_complex "{vstack_filter}" -c:v libx264 -preset fast -crf 23 -y "{temp_merged}"'
    print("垂直拼接各行...")
    os.system(cmd_merge)
    
    # 添加上下黑边到1920x1080（当前是1920x960）
    print("添加黑边到1080p...")
    cmd_pad = f'ffmpeg -i "{temp_merged}" -vf "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black" -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k -y "{output_path}"'
    os.system(cmd_pad)
    
    # 清理临时文件
    print("\n清理临时文件...")
    for vp in processed_videos:
        try:
            os.remove(vp)
        except:
            pass
    for rv in row_videos:
        try:
            os.remove(rv)
        except:
            pass
    try:
        os.remove(temp_merged)
    except:
        pass
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    print(f"\n完成！输出文件: {output_path}")

if __name__ == "__main__":
    main()