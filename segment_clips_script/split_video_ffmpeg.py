import os
import subprocess

# 输入文件夹路径（包含所有视频）
input_folder = "/home/zikaixiao/zikaixiao/VideoMAEv2/data_factory/RAW_320/EndoVis2019/Videos/Full/standard"

# 输出根文件夹路径
output_root_folder = "/home/zikaixiao/zikaixiao/VideoMAEv2/data_factory/PT_clips/EndoVis2019"

# 每个片段的时长（秒）
clip_duration = 10

# 支持的视频文件扩展名
video_extensions = (".mp4", ".avi", ".mov", ".mkv")

# 遍历输入文件夹中的所有视频文件
for root, dirs, files in os.walk(input_folder):
    for file in files:
        # 检查文件扩展名是否为视频文件
        if file.lower().endswith(video_extensions):
            # 输入视频文件的完整路径
            input_video_path = os.path.join(root, file)
            
            # 获取视频文件名（不带扩展名）
            video_name = os.path.splitext(file)[0]
            
            # 创建对应的输出文件夹
            output_folder = os.path.join(output_root_folder, video_name + "_clips")
            os.makedirs(output_folder, exist_ok=True)
            
            # 使用 ffmpeg 分割视频
            command = [
                "ffmpeg",
                "-i", input_video_path,          # 输入视频文件
                "-c", "copy",                    # 直接复制流，不重新编码
                "-f", "segment",                 # 分段输出模式
                "-segment_time", str(clip_duration),  # 每个片段的时长
                "-reset_timestamps", "1",        # 重置时间戳
                "-map", "0",                     # 选择所有流（视频、音频等）
                os.path.join(output_folder, f"{video_name}_clip%05d.mp4")  # 输出文件名格式
            ]
            
            # 打印当前处理的视频文件
            print(f"Processing: {input_video_path}")
            
            # 运行 ffmpeg 命令
            try:
                subprocess.run(command, check=True)
                print(f"Finished splitting: {input_video_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error splitting {input_video_path}: {e}")
            
            print("-" * 50)

print("All videos have been processed.")