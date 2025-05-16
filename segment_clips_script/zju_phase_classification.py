"""
文件功能：
该脚本用于处理 ZJU Phase Classification 数据集中的视频文件，具体功能包括：
1. 将原始视频切割为固定时长（默认 30 秒）的片段。
2. 对每个片段进行加速处理（默认加速 3 倍，压缩为 10 秒）。
3. 生成对应的元数据文件，记录每个片段的路径、标签和时长等信息。

使用方式：
1. 修改 `RAW_DATA_ROOT` 变量，将其设置为原始视频数据的根目录路径。
2. 修改 `OUTPUT_CLIPS_DIR` 变量，设置输出视频片段的目录路径。
3. 修改 `METADATA_PATH` 变量，设置元数据文件的保存路径。
4. 可选：调整 `CLIP_DURATION` 和 `SPEED_FACTOR` 变量，修改片段时长和加速倍数。
5. 运行脚本，脚本会自动处理视频并生成元数据文件。

注意事项：
- 原始视频文件应按照类别存放在 `RAW_DATA_ROOT` 的子目录中。
- 每个视频文件会被切割为固定时长的片段，并根据加速倍数压缩时长。
- 如果输出目录已存在，脚本会跳过已处理的片段。
- 需要安装 `ffmpeg` 和 `ffprobe` 工具以处理视频文件。

示例：
假设原始数据目录结构如下：
/RAW/zju_phase/
    ├── Category1/
        ├── video1.mp4
        ├── video2.mp4
    ├── Category2/
        ├── video3.mp4

运行脚本后，输出目录结构如下：
/SFT/zju_phase_classification/clips/
    ├── video1_clip000.mp4
    ├── video1_clip001.mp4
    ├── video2_clip000.mp4
    ├── video3_clip000.mp4
/SFT/zju_phase_classification/zju_phase_classification_info.json
"""

import os
import json
import subprocess
from tqdm import tqdm

# ===================== 配置参数 =====================
RAW_DATA_ROOT = ""  # 原始数据根目录
OUTPUT_CLIPS_DIR = ""  # 输出片段目录
METADATA_PATH = ""  # 元数据文件路径
CLIP_DURATION = 30  # 原始片段时长（秒）
SPEED_FACTOR = 3    # 加速倍数（30/3=10秒）

# ===================== 函数定义 =====================

def get_video_duration(video_path):
    """使用ffprobe获取视频精确时长（浮点数秒）"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=nw=1:nk=1',
            video_path
        ]
        output = subprocess.check_output(cmd).decode().strip()
        return float(output)
    except Exception as e:
        print(f"获取时长失败: {video_path} - {str(e)}")
        return None

def has_audio_stream(video_path):
    """检测视频是否包含音频流"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0',
            video_path
        ]
        return subprocess.check_output(cmd).decode().strip() != ''
    except:
        return False

def generate_clips(video_path, output_dir, clip_duration=30, speed_factor=3):
    """
    生成加速视频片段
    参数:
        video_path: 原始视频路径
        output_dir: 输出目录
        clip_duration: 原始片段时长（秒）
        speed_factor: 加速倍数
    返回:
        生成的片段信息列表（包含起始时间和持续时间的元组）
    """
    # 获取视频总时长
    total_duration = get_video_duration(video_path)
    if total_duration is None:
        return []

    # 生成时间分段（前闭后开区间）
    segments = []
    current_start = 0.0
    while current_start < total_duration:
        # 计算实际片段时长（不超过剩余时长）
        actual_duration = min(clip_duration, total_duration - current_start)
        segments.append((
            current_start,        # 起始时间
            actual_duration,      # 实际持续时间
            actual_duration / speed_factor  # 压缩后时长
        ))
        current_start += clip_duration

    # 处理每个片段
    processed_segments = []
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    has_audio = has_audio_stream(video_path)

    for idx, (start, orig_duration, compressed_duration) in enumerate(segments):
        output_filename = f"{base_name}_clip{idx:03d}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        # 跳过已存在的文件
        if os.path.exists(output_path):
            continue

        # 构建FFmpeg命令（单滤镜链同时处理视频和音频）
        filter_chain = []
        maps = []

        # 视频处理：加速
        filter_chain.append(f"[0:v]setpts={1/speed_factor}*PTS[vout]")
        maps.extend(['-map', '[vout]'])

        # 音频处理（如果存在）
        if has_audio:
            filter_chain.append(f"[0:a]atempo={speed_factor}[aout]")
            maps.extend(['-map', '[aout]'])

        try:
            cmd = [
                'ffmpeg', '-y',
                '-ss', f"{start:.3f}",  # 精确到毫秒级定位
                '-i', video_path,
                '-t', f"{orig_duration:.3f}",  # 精确截取时长
                '-filter_complex', ';'.join(filter_chain),
                *maps,
                '-c:v', 'libx264', '-preset', 'fast',  # 快速编码预设
                '-c:a', 'aac' if has_audio else '-an',
                '-movflags', '+faststart',  # 优化网络播放
                output_path
            ]
            
            # 执行命令（隐藏输出）
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # 记录成功处理的片段
            processed_segments.append({
                'path': output_path,
                'compressed_duration': compressed_duration
            })
        except Exception as e:
            print(f"片段生成失败: {output_path} - {str(e)}")

    return processed_segments

# ===================== 主处理流程 =====================

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)

    # 获取分类目录（自动排序）
    categories = sorted([
        f for f in os.listdir(RAW_DATA_ROOT)
        if os.path.isdir(os.path.join(RAW_DATA_ROOT, f))
    ])
    label_map = {name: idx for idx, name in enumerate(categories)}

    metadata = []

    # 遍历每个类别目录
    for category in categories:
        print(f"\n处理类别: {category} ({label_map[category]})")
        category_dir = os.path.join(RAW_DATA_ROOT, category)
        videos = [f for f in os.listdir(category_dir) if f.endswith('.mp4')]

        # 进度条显示
        pbar = tqdm(videos, desc=f"{category[:15]:<15}", unit="video")
        
        for video_file in pbar:
            video_path = os.path.join(category_dir, video_file)
            
            # 生成并处理视频片段
            clips = generate_clips(
                video_path=video_path,
                output_dir=OUTPUT_CLIPS_DIR,
                clip_duration=CLIP_DURATION,
                speed_factor=SPEED_FACTOR
            )

            # 记录元数据
            for clip in clips:
                rel_path = os.path.relpath(clip['path'], start=os.path.dirname(OUTPUT_CLIPS_DIR))
                metadata.append({
                    "label": label_map[category],
                    "label_description": category,
                    "base_path": "",  # 使用相对路径
                    "relative_path": "zju_phase_classification/clips/" + rel_path,
                    "duration": round(clip['compressed_duration'], 2),
                    "dataset_name": "zju_phase_classification",
                    "task_type": "phase_classification"
                })

    # 保存元数据文件
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n处理完成！共生成 {len(metadata)} 个视频片段")
    print(f"元数据文件已保存至: {METADATA_PATH}")

if __name__ == "__main__":
    main()
