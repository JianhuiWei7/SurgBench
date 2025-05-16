"""
文件功能：
该脚本用于处理 AutoLaparo 数据集中的视频和标签数据，生成统一的 JSON 配置文件。
具体功能包括：
1. 处理任务1（手术阶段识别）：
   - 从原始视频中按固定帧数分割视频片段。
   - 根据标签文件为每个片段分配主标签。
   - 将片段保存到指定目录，并生成对应的元数据。
2. 处理任务2（运动预测）：
   - 从原始片段中复制视频文件。
   - 根据标签文件为每个片段分配标签。
   - 将片段保存到指定目录，并生成对应的元数据。
3. 将两个任务的结果合并为一个 JSON 文件，包含所有片段的路径、标签和元数据。

使用方式：
1. 修改 `BASE_DIR` 变量，将其设置为数据工厂的根目录路径。
2. 确保输入路径（如 `TASK1_VIDEO_DIR`、`TASK2_CLIP_DIR` 等）正确指向原始数据。
3. 修改 `OUTPUT_CLIP_DIR` 和 `OUTPUT_JSON_PATH` 变量，设置输出片段和 JSON 文件的路径。
4. 运行脚本，脚本会自动处理数据并生成 JSON 配置文件。

注意事项：
- 任务1的视频片段会根据 `CLIP_LENGTH` 参数按固定帧数分割。
- 任务2的视频片段会直接从原始片段复制。
- 标签映射（`PHASE_MAPPING` 和 `MOTION_MAPPING`）用于将原始标签转换为统一编号。
- 需要安装 `ffmpeg` 和 `ffprobe` 工具以处理视频文件。

示例：
假设输入目录结构如下：
/RAW/AutoLaparo/task1/videos/
    ├── 1.mp4
    ├── 2.mp4
/RAW/AutoLaparo/task1/labels/
    ├── label_1.txt
    ├── label_2.txt
/RAW/AutoLaparo/task2/clips/
    ├── 001.mp4
    ├── 002.mp4

运行脚本后，输出目录结构如下：
/SFT/AutoLaparo/clips/
    ├── task1_v1_clip0000.mp4
    ├── task1_v1_clip0001.mp4
    ├── task2_001.mp4
    ├── task2_002.mp4
/SFT/AutoLaparo/AutoLaparo_info.json
"""

import os
import subprocess
import json
import pandas as pd
from collections import Counter

# ======================== 配置参数 ========================
BASE_DIR = "/home/zikaixiao/zikaixiao/VideoMAEv2/data_factory"
RAW_DATA_DIR = os.path.join(BASE_DIR, "RAW/AutoLaparo")
SFT_DATA_DIR = os.path.join(BASE_DIR, "SFT/AutoLaparo")

# 输入路径
TASK1_VIDEO_DIR = os.path.join(RAW_DATA_DIR, "task1/videos")
TASK1_LABEL_DIR = os.path.join(RAW_DATA_DIR, "task1/labels")
TASK2_CLIP_DIR = os.path.join(RAW_DATA_DIR, "task2/clips")
TASK2_LABEL_PATH = os.path.join(RAW_DATA_DIR, "task2/laparoscope_motion_label.txt")

# 输出路径
OUTPUT_CLIP_DIR = os.path.join(SFT_DATA_DIR, "clips")
os.makedirs(OUTPUT_CLIP_DIR, exist_ok=True)
OUTPUT_JSON_PATH = os.path.join(SFT_DATA_DIR, "AutoLaparo_info.json")

# ======================== 标签映射 ========================
# 任务1标签映射（Phase -> 统一编号）及每个类别的剪辑长度（帧数）
PHASE_CONFIG = {
    1: {"unified_label": 0, "clip_length": 20, "description": "Preparation"},
    2: {"unified_label": 1, "clip_length": 50, "description": "Dividing Ligament and Peritoneum"},
    3: {"unified_label": 2, "clip_length": 50, "description": "Dividing Uterine Vessels and Ligament"},
    4: {"unified_label": 3, "clip_length": 50, "description": "Transecting the Vagina"},
    5: {"unified_label": 4, "clip_length": 10, "description": "Specimen Removal"},
    6: {"unified_label": 5, "clip_length": 50, "description": "Suturing"},
    7: {"unified_label": 6, "clip_length": 50, "description": "Washing"}
}

# 任务2标签映射（Motion -> 统一编号从7开始）及每个类别的剪辑长度（帧数）
MOTION_CONFIG = {
    0: {"unified_label": 7, "clip_length": 50, "description": "Static"},
    1: {"unified_label": 8, "clip_length": 50, "description": "Up"},
    2: {"unified_label": 9, "clip_length": 50, "description": "Down"},
    3: {"unified_label": 10, "clip_length": 50, "description": "Left"},
    4: {"unified_label": 11, "clip_length": 50, "description": "Right"},
    5: {"unified_label": 12, "clip_length": 50, "description": "Zoom-in"},
    6: {"unified_label": 13, "clip_length": 50, "description": "Zoom-out"}
}

# ======================== 工具函数 ========================
def get_video_duration(video_path):
    """获取视频时长（秒）"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def get_video_frame_rate(video_path):
    """获取视频帧率"""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    frame_rate_str = result.stdout.strip()
    numerator, denominator = map(int, frame_rate_str.split('/'))
    return numerator / denominator

# ======================== 任务1处理 ========================
def process_task1():
    entries = []
    
    for video_file in sorted(os.listdir(TASK1_VIDEO_DIR)):
        if not video_file.endswith(".mp4"):
            continue
        
        # 解析视频编号
        video_num = str(os.path.splitext(video_file)[0])
        video_path = os.path.join(TASK1_VIDEO_DIR, video_file)
        label_path = os.path.join(TASK1_LABEL_DIR, f"label_{video_num}.txt")
        
        print(f"\nProcessing Task1 video {video_num}...")
        
        # 获取视频帧率
        try:
            frame_rate = get_video_frame_rate(video_path)
        except Exception as e:
            print(f"Error getting frame rate for {video_path}: {str(e)}")
            continue
        
        # 读取标签数据
        try:
            df = pd.read_csv(label_path, sep='\t', skiprows=1, names=['Frame', 'Phase'])
            phases = df['Phase'].tolist()
        except Exception as e:
            print(f"Error reading label file {label_path}: {str(e)}")
            continue
        
        # 按类别分割视频
        current_phase = None
        start_frame = 0
        clip_counter = 0
        
        for frame_idx, phase in enumerate(phases):
            if phase != current_phase:
                # 处理上一个phase的剩余帧
                if current_phase is not None:
                    phase_config = PHASE_CONFIG.get(current_phase)
                    if phase_config:
                        clip_length = phase_config["clip_length"]
                        # 分割剩余的帧
                        while start_frame < frame_idx:
                            end_frame = min(start_frame + clip_length, frame_idx)
                            if end_frame - start_frame < clip_length // 2:  # 如果剩余帧太少，合并到上一个片段
                                break
                            
                            # 生成剪辑
                            output_name = f"task1_v{video_num}_{PHASE_CONFIG[current_phase]['description'].replace(' ', '_')}_clip{clip_counter:04d}.mp4"
                            output_path = os.path.join(OUTPUT_CLIP_DIR, output_name)
                            
                            if not os.path.exists(output_path):
                                start_time = start_frame / frame_rate
                                duration = (end_frame - start_frame) / frame_rate
                                
                                try:
                                    subprocess.run([
                                        'ffmpeg', '-y',
                                        '-ss', str(start_time),
                                        '-i', video_path,
                                        '-t', str(duration),
                                        '-c', 'copy',
                                        output_path
                                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                                except Exception as e:
                                    print(f"Error processing clip {output_name}: {str(e)}")
                                    start_frame = end_frame
                                    continue
                            
                            duration = (end_frame - start_frame) / frame_rate
                            entries.append({
                                "label": PHASE_CONFIG[current_phase]["unified_label"],
                                "label_description": PHASE_CONFIG[current_phase]["description"],
                                "base_path": SFT_DATA_DIR,
                                "relative_path": f"clips/{output_name}",
                                "duration": round(duration, 2),
                                "dataset_name": "AutoLaparo",
                                "task_type": "surgical_phase"
                            })
                            clip_counter += 1
                            start_frame = end_frame
                
                # 开始新的phase
                current_phase = phase
                start_frame = frame_idx
        
        # 处理最后一个phase
        if current_phase is not None and start_frame < len(phases):
            phase_config = PHASE_CONFIG.get(current_phase)
            if phase_config:
                clip_length = phase_config["clip_length"]
                while start_frame < len(phases):
                    end_frame = min(start_frame + clip_length, len(phases))
                    if end_frame - start_frame < clip_length // 2:  # 如果剩余帧太少，合并到上一个片段
                        break
                    
                    output_name = f"task1_v{video_num}_{PHASE_CONFIG[current_phase]['description'].replace(' ', '_')}_clip{clip_counter:04d}.mp4"
                    output_path = os.path.join(OUTPUT_CLIP_DIR, output_name)
                    
                    if not os.path.exists(output_path):
                        start_time = start_frame / frame_rate
                        duration = (end_frame - start_frame) / frame_rate
                        
                        try:
                            subprocess.run([
                                'ffmpeg', '-y',
                                '-ss', str(start_time),
                                '-i', video_path,
                                '-t', str(duration),
                                '-c', 'copy',
                                output_path
                            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                        except Exception as e:
                            print(f"Error processing clip {output_name}: {str(e)}")
                            start_frame = end_frame
                            continue
                    
                    duration = (end_frame - start_frame) / frame_rate
                    entries.append({
                        "label": PHASE_CONFIG[current_phase]["unified_label"],
                        "label_description": PHASE_CONFIG[current_phase]["description"],
                        "base_path": SFT_DATA_DIR,
                        "relative_path": f"clips/{output_name}",
                        "duration": round(duration, 2),
                        "dataset_name": "AutoLaparo",
                        "task_type": "surgical_phase"
                    })
                    clip_counter += 1
                    start_frame = end_frame
    
    return entries

# ======================== 任务2处理 ========================
def process_task2():
    entries = []
    
    try:
        df = pd.read_csv(TASK2_LABEL_PATH, sep='\t', skiprows=1, names=['Clip', 'Label', 'Phase'])
    except Exception as e:
        print(f"Error reading task2 label file: {str(e)}")
        return []
    
    for _, row in df.iterrows():
        clip_num = str(row['Clip']).zfill(3)
        original_label = int(row['Label'])
        
        # 获取配置
        motion_config = MOTION_CONFIG.get(original_label)
        if not motion_config:
            continue
        
        # 输入路径
        input_path = os.path.join(TASK2_CLIP_DIR, f"{clip_num}.mp4")
        if not os.path.exists(input_path):
            print(f"Source file not found: {input_path}")
            continue
        
        # 获取视频信息
        try:
            frame_rate = get_video_frame_rate(input_path)
            duration = get_video_duration(input_path)
            total_frames = int(duration * frame_rate)
        except Exception as e:
            print(f"Error getting video info for {input_path}: {str(e)}")
            continue
        
        # 根据clip_length分割视频
        clip_length = motion_config["clip_length"]
        clip_counter = 0
        
        for start_frame in range(0, total_frames, clip_length):
            end_frame = min(start_frame + clip_length, total_frames)
            
            # 如果剩余帧数不足clip_length的一半，合并到上一个片段
            if (total_frames - end_frame) < clip_length // 2:
                end_frame = total_frames
            
            # 跳过太短的片段
            if (end_frame - start_frame) < clip_length // 2:
                continue
            
            # 生成输出文件名
            output_name = f"task2_{motion_config['description'].replace(' ', '_')}_{clip_num}_clip{clip_counter:04d}.mp4"
            output_path = os.path.join(OUTPUT_CLIP_DIR, output_name)
            
            # 分割视频
            if not os.path.exists(output_path):
                start_time = start_frame / frame_rate
                duration = (end_frame - start_frame) / frame_rate
                
                try:
                    subprocess.run([
                        'ffmpeg', '-y',
                        '-ss', str(start_time),
                        '-i', input_path,
                        '-t', str(duration),
                        '-c', 'copy',
                        output_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                except Exception as e:
                    print(f"Error processing clip {output_name}: {str(e)}")
                    continue
            
            # 计算实际时长
            clip_duration = (end_frame - start_frame) / frame_rate
            
            entries.append({
                "label": motion_config["unified_label"],
                "label_description": motion_config["description"],
                "base_path": SFT_DATA_DIR,
                "relative_path": f"clips/{output_name}",
                "duration": round(clip_duration, 2),
                "dataset_name": "AutoLaparo",
                "task_type": "motion_prediction"
            })
            
            clip_counter += 1
    
    return entries

# ======================== 主程序 ========================
if __name__ == "__main__":
    # 处理两个任务
    task1_entries = process_task1()
    print(f"\nProcessed {len(task1_entries)} Task1 entries")
    
    task2_entries = process_task2()
    print(f"Processed {len(task2_entries)} Task2 entries")
    
    # 合并结果
    all_entries = task1_entries + task2_entries
    
    # 保存JSON
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(all_entries, f, indent=2)
    
    print(f"\nTotal {len(all_entries)} entries saved to {OUTPUT_JSON_PATH}")
    print("Processing completed!")