# ==============================================================================
# 脚本功能：
# 本脚本用于处理 Cholec80 数据集中的原始手术视频。主要功能包括：
# 1. 视频分段：将长视频按固定帧数（CLIP_FRAMES）切割成较短的视频片段（clips）。
#    - 每个片段有最小帧数要求（MIN_FRAMES）。
#    - 如果视频尾部剩余帧数不足 MIN_FRAMES，则舍弃该尾部。
# 2. 标签管理：从原始标注文件中提取手术阶段（phase）和手术工具（tool）的标签，
#    并为它们创建唯一的数字索引。
# 3. 标签分析：对于每个生成的视频片段：
#    - 分析该片段内最主要的手术阶段，并计算其在该片段中的置信度（出现帧数比例）。
#    - 分析该片段内出现的所有手术工具，并计算各工具在该片段中的置信度（出现帧数比例）。
# 4. JSON元数据生成：为每个有效的（片段帧数 >= MIN_FRAMES）且带有标签的片段生成一个 JSON 条目，
#    包含片段路径、时长、帧数、数据集名称、任务类型（阶段分类/工具分类）、标签信息和置信度。
# 5. 数量限制与筛选：
#    - 总共最多生成 MAX_ENTRIES 个 JSON 条目。
#    - 如果生成的潜在条目总数超过 MAX_ENTRIES，则根据置信度从高到低进行排序。
#    - 如果在 MAX_ENTRIES 的边界处存在置信度相同的条目，则从这些置信度相同的条目中随机选择，以填满 MAX_ENTRIES。
# 6. 输出：
#    - 切割后的视频片段保存在 `output_clip_dir` 目录下。
#    - 最终筛选后的 JSON 元数据保存在 `base_path` 下的 `cholec80_info.json` 文件中。
#
# 主要参数：
# - base_path: 项目基础路径，用于存放输出的 JSON 文件和 clips 文件夹的父目录。
# - raw_video_root: 原始 Cholec80 数据集根目录，包含视频、阶段标注和工具标注。
# - output_clip_dir: 切割后视频片段的输出目录。
# - MIN_FRAMES: 生成的视频片段所需的最少帧数。
# - CLIP_FRAMES: 目标切割片段的帧数。
# - MAX_ENTRIES: 最终 JSON 文件中包含的最大条目数量。
#
# 使用方法：
# 1. 配置 `base_path` 和 `raw_video_root` 指向正确的路径。
# 2. 根据需要调整 `MIN_FRAMES`, `CLIP_FRAMES`, `MAX_ENTRIES`。
# 3. 运行此脚本。
#
# 注意事项：
# - 确保已安装 `opencv-python` (cv2) 和 `ffmpeg`。
# - 原始视频文件、阶段标注文件和工具标注文件需要按 Cholec80 数据集的标准结构存放。
# - 脚本会创建 `output_clip_dir` 目录（如果不存在）。
# ==============================================================================

import os
import csv
import json
import cv2
import subprocess
from collections import defaultdict
import random # For random selection in ties

# =================配置参数=================
base_path = "/home/zikaixiao/zikaixiao/VideoMAEv2/data_factory/SFT" # SFT directory
raw_video_root = "/home/zikaixiao/zikaixiao/VideoMAEv2/data_factory/RAW_320/cholec80" # 请确保此路径正确
output_clip_dir = os.path.join(base_path, "cholec80/clips") # Clips will be in SFT/cholec80/clips
os.makedirs(output_clip_dir, exist_ok=True)

MIN_FRAMES = 80
CLIP_FRAMES = 300
MAX_ENTRIES = 5000

# =================标签管理=================
class LabelManager:
    def __init__(self):
        self.phase_labels = self._get_phase_labels()
        self.tool_labels = self._get_tool_labels(len(self.phase_labels))
        
    def _get_phase_labels(self):
        phases = set()
        for vid in range(1, 81):
            file_path = os.path.join(raw_video_root, "phase_annotations", f"video{vid:02d}-phase.txt")
            if not os.path.exists(file_path):
                print(f"警告: 阶段标注文件未找到: {file_path}")
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                try:
                    next(reader)
                except StopIteration:
                    print(f"警告: 阶段标注文件为空或只有表头: {file_path}")
                    continue
                phases.update(row[1] for row in reader if len(row) >= 2)
        if not phases:
            print("错误: 未找到任何阶段标签。请检查 raw_video_root 和标注文件路径。")
            return {}
        return {phase: i for i, phase in enumerate(sorted(list(phases)))}
    
    def _get_tool_labels(self, start_idx):
        sample_file = os.path.join(raw_video_root, "tool_annotations", "video01-tool.txt")
        if not os.path.exists(sample_file):
            print(f"警告: 示例工具标注文件未找到: {sample_file}。无法确定工具标签。")
            return {}
        with open(sample_file, 'r', encoding='utf-8') as f:
            try:
                tools = f.readline().strip().split('\t')[1:]
            except IndexError:
                print(f"警告: 示例工具标注文件格式不正确或为空: {sample_file}")
                return {}
        return {tool: start_idx + i for i, tool in enumerate(tools)}

# =================视频处理=================
def process_clip(video_path, clip_info, label_mgr):
    if not os.path.exists(video_path):
        print(f"错误: 视频文件未找到: {video_path}")
        return []
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {video_path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps == 0: 
        print(f"错误: 视频 {video_path} 的 FPS 为 0 或无法读取。跳过处理。")
        return []
    if total_frames == 0:
        print(f"错误: 视频 {video_path} 的 总帧数为 0 或无法读取。跳过处理。")
        return []
    
    entries = [] 
    clip_idx = 0 
    
    start_frame = 0
    while start_frame < total_frames:
        end_frame = min(start_frame + CLIP_FRAMES - 1, total_frames - 1)
        current_segment_frame_count = end_frame - start_frame + 1

        if current_segment_frame_count < MIN_FRAMES:
            if clip_idx == 0 and total_frames >= MIN_FRAMES : 
                end_frame = total_frames - 1
                current_segment_frame_count = end_frame - start_frame + 1
                if current_segment_frame_count < MIN_FRAMES:
                    break 
            else: 
                break 

        start_time = start_frame / fps
        
        clip_filename_no_ext = f"{os.path.basename(video_path).split('.')[0]}_{start_frame:06d}_{end_frame:06d}"
        clip_filename = f"{clip_filename_no_ext}.mp4"
        clip_path = os.path.join(output_clip_dir, clip_filename)
        
        if not os.path.exists(clip_path):
            ffmpeg_command = [
                'ffmpeg', '-y', 
                '-ss', str(start_time),
                '-i', video_path,
                '-frames:v', str(current_segment_frame_count), 
                '-c', 'copy',
                '-an', # No audio
                '-loglevel', 'error', # Suppress verbose output, only errors
                clip_path
            ]
            try:
                subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except subprocess.CalledProcessError as e:
                print(f"ffmpeg 处理 {clip_filename} 时出错: {e.stderr.decode()}")
                start_frame += CLIP_FRAMES
                continue
        
        actual_frames = verify_clip_frames(clip_path)
        if actual_frames < MIN_FRAMES:
            # print(f"警告: 片段 {clip_filename} 实际生成 {actual_frames} 帧 (少于 {MIN_FRAMES} 帧)，将被跳过并删除。")
            if os.path.exists(clip_path):
                try:
                    os.remove(clip_path)
                except OSError as e:
                    print(f"警告: 删除文件 {clip_path} 失败: {e}")
            if clip_idx == 0 and (start_frame + CLIP_FRAMES >= total_frames):
                 break
            start_frame += CLIP_FRAMES 
            continue
        
        actual_duration = actual_frames / fps 

        phase_entry_data = analyze_phase(clip_info['phase'], start_frame, end_frame, label_mgr)
        tool_entries_data = analyze_tools(clip_info['tools'], start_frame, end_frame, label_mgr)
        
        # relative_path should be like "cholec80/clips/video01_000000_000299.mp4"
        # base_path is SFT directory. output_clip_dir is SFT/cholec80/clips
        # os.path.relpath(clip_path, base_path) would give cholec80/clips/filename.mp4
        relative_clip_path = os.path.relpath(clip_path, base_path)

        base_entry = {
            "base_path": base_path,
            "relative_path": relative_clip_path,
            "duration": round(actual_duration, 2),
            "dataset_name": "cholec80",
            "frame_count": actual_frames
        }
        
        if phase_entry_data:
            entries.append({**base_entry, **phase_entry_data})
        for tool_data in tool_entries_data:
            entries.append({**base_entry, **tool_data})
        
        clip_idx += 1
        if clip_idx == 1 and total_frames < CLIP_FRAMES and total_frames >= MIN_FRAMES:
            break 
        start_frame += CLIP_FRAMES
    
    return entries

def verify_clip_frames(clip_path):
    if not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
        return 0
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        # print(f"警告: 无法打开验证的片段 {clip_path}。返回 0 帧。")
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def analyze_phase(phase_data, start_frame, end_frame, label_mgr):
    phase_counter = defaultdict(int)
    annotated_frames_in_segment = 0 
    
    effective_end_frame = min(end_frame, len(phase_data) - 1)
    if start_frame > effective_end_frame : # No valid frames in this range for phase_data
        return None

    for frame_idx in range(start_frame, effective_end_frame + 1):
        current_phase = phase_data[frame_idx]
        phase_counter[current_phase] += 1
        annotated_frames_in_segment += 1
    
    if not phase_counter or annotated_frames_in_segment == 0:
        return None
        
    dominant_phase = max(phase_counter, key=phase_counter.get)
    max_count = phase_counter[dominant_phase]
    confidence = max_count / annotated_frames_in_segment if annotated_frames_in_segment > 0 else 0.0
    
    if dominant_phase not in label_mgr.phase_labels:
        print(f"警告: 主要阶段 '{dominant_phase}' (帧 {start_frame}-{end_frame}) 不在 phase_labels 中。跳过此阶段条目。")
        return None

    return {
        "label": label_mgr.phase_labels[dominant_phase],
        "label_index": str(label_mgr.phase_labels[dominant_phase]),
        "label_description": dominant_phase,
        "task_type": "phase_classification",
        "confidence": round(confidence, 4)
    }

def analyze_tools(tool_data, start_frame, end_frame, label_mgr):
    tool_activity_counts = defaultdict(int) 
    
    effective_end_frame = min(end_frame, len(tool_data) - 1)
    if start_frame > effective_end_frame: # No valid frames in this range for tool_data
        return []
        
    frames_in_segment_for_tool_analysis = (effective_end_frame - start_frame + 1)
    
    if frames_in_segment_for_tool_analysis <= 0:
        return []

    for frame_idx in range(start_frame, effective_end_frame + 1):
        frame_tools_status = tool_data[frame_idx]
        for tool_name, is_present in frame_tools_status.items():
            if is_present: 
                tool_activity_counts[tool_name] += 1
    
    entries = []
    if not tool_activity_counts:
        return entries

    for tool_name, count in tool_activity_counts.items():
        confidence = count / frames_in_segment_for_tool_analysis if frames_in_segment_for_tool_analysis > 0 else 0.0
        
        if tool_name not in label_mgr.tool_labels:
            print(f"警告: 工具 '{tool_name}' (帧 {start_frame}-{end_frame}) 不在 tool_labels 中。跳过此工具条目。")
            continue

        entries.append({
            "label": label_mgr.tool_labels[tool_name],
            "label_index": str(label_mgr.tool_labels[tool_name]),
            "label_description": tool_name,
            "task_type": "tool_classification",
            "confidence": round(confidence, 4)
        })
    return entries

# =================主程序=================
if __name__ == "__main__":
    label_mgr = LabelManager()
    if not label_mgr.phase_labels or not label_mgr.tool_labels:
        print("错误: 标签管理器未能初始化标签。程序退出。")
        exit()

    all_potential_entries = []
    for vid_num in range(1, 81):
        video_name = f"video{vid_num:02d}"
        print(f"正在处理视频: {video_name}...")
        
        phase_file = os.path.join(raw_video_root, "phase_annotations", f"{video_name}-phase.txt")
        phase_data = []
        if not os.path.exists(phase_file):
            print(f"  阶段标注文件未找到: {phase_file}。此视频的阶段信息将为空。")
        else:
            with open(phase_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                try:
                    next(reader) 
                    phase_data = [row[1] for row in reader if len(row) >=2]
                except StopIteration:
                    print(f"  阶段标注文件为空或只有表头: {phase_file}")
                if not phase_data:
                     print(f"  未从 {phase_file} 加载到阶段数据。")

        tool_file = os.path.join(raw_video_root, "tool_annotations", f"{video_name}-tool.txt")
        tool_data_for_video = []
        
        # Initialize with all globally known tools set to 0 for every frame by default
        # This ensures tool_data_for_video[frame_idx] will have all tool keys
        fallback_tool_status = {tool_name: 0 for tool_name in label_mgr.tool_labels.keys()}

        if not os.path.exists(tool_file):
            print(f"  工具标注文件未找到: {tool_file}。")
            if phase_data: # If phase data exists, fill tool data with fallbacks
                 print(f"  将使用全0工具状态填充至阶段数据长度 ({len(phase_data)} 帧)。")
                 tool_data_for_video = [fallback_tool_status.copy() for _ in range(len(phase_data))]
        else:
            with open(tool_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                try:
                    video_specific_tool_headers = next(reader)[1:]
                except (StopIteration, IndexError):
                    print(f"  工具标注文件表头格式不正确或为空: {tool_file}。将使用全0工具状态。")
                    video_specific_tool_headers = []
                
                if video_specific_tool_headers:
                    valid_headers_for_this_video = []
                    header_to_original_idx_map = {} # Maps valid tool name to its column index in current file
                    for idx, tool_h in enumerate(video_specific_tool_headers):
                        if tool_h in label_mgr.tool_labels:
                            valid_headers_for_this_video.append(tool_h)
                            header_to_original_idx_map[tool_h] = idx
                        else:
                            print(f"警告: {video_name} 工具 '{tool_h}' 不在全局工具列表，将被忽略。")
                    
                    last_frame_num = -1
                    for row_idx, row in enumerate(reader):
                        if len(row) < (len(video_specific_tool_headers) + 1): # Frame col + tool cols
                            continue
                        try:
                            frame_num = int(row[0])
                            current_frame_status = fallback_tool_status.copy() # Start with all tools 0
                            for tool_name in valid_headers_for_this_video:
                                original_col_idx = header_to_original_idx_map[tool_name]
                                current_frame_status[tool_name] = int(row[original_col_idx + 1])
                        except ValueError:
                            # print(f"  跳过 {tool_file} 中包含非整数值的行 {row_idx+2}: {row}")
                            continue

                        # Fill missing frames with last known status
                        fill_status = tool_data_for_video[-1].copy() if tool_data_for_video else fallback_tool_status.copy()
                        for _ in range(last_frame_num + 1, frame_num):
                            tool_data_for_video.append(fill_status)
                        
                        tool_data_for_video.append(current_frame_status)
                        last_frame_num = frame_num
                else: # No valid headers or empty tool file content after header
                    if phase_data:
                        print(f"  视频 {video_name} 工具文件无有效表头/内容，将使用全0工具状态填充。")
                        tool_data_for_video = [fallback_tool_status.copy() for _ in range(len(phase_data))]
        
        # Align tool_data_for_video length with phase_data length (master length)
        num_phase_frames = len(phase_data)
        if num_phase_frames > 0: # Only align if phase_data is present
            current_tool_frames = len(tool_data_for_video)
            if current_tool_frames < num_phase_frames:
                # print(f"  填充工具数据以匹配阶段数据长度 (从 {current_tool_frames} 到 {num_phase_frames} 帧)。")
                last_tool_status = tool_data_for_video[-1].copy() if tool_data_for_video else fallback_tool_status.copy()
                tool_data_for_video.extend([last_tool_status.copy()] * (num_phase_frames - current_tool_frames))
            elif current_tool_frames > num_phase_frames:
                # print(f"  截断工具数据以匹配阶段数据长度 (从 {current_tool_frames} 到 {num_phase_frames} 帧)。")
                tool_data_for_video = tool_data_for_video[:num_phase_frames]
        # If num_phase_frames is 0, tool_data_for_video might still have data if tool file was longer.
        # analyze_phase will return None. analyze_tools will use len(tool_data_for_video).
        # process_clip uses total_frames from video for cutting. This is fine.

        video_path = os.path.join(raw_video_root, "videos", f"{video_name}.mp4")
        if not os.path.exists(video_path):
            print(f"  视频文件未找到: {video_path}。跳过此视频。")
            continue
        
        # If both annotation types are empty, processing the video might not yield labeled clips.
        if not phase_data and not tool_data_for_video:
             # Check if video itself is valid before printing this specific warning
            temp_cap = cv2.VideoCapture(video_path)
            if temp_cap.isOpened() and temp_cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                print(f"  警告: 视频 {video_name} 无有效的阶段或工具标注数据。仍会尝试切割视频，但可能不会生成JSON条目。")
            temp_cap.release()


        clip_entries = process_clip(video_path, {
            'phase': phase_data,
            'tools': tool_data_for_video
        }, label_mgr)
        
        all_potential_entries.extend(clip_entries)
        print(f"  完成处理 {video_name}.mp4, 生成了 {len(clip_entries)} 个潜在 JSON 条目。")

    print(f"\n总共生成了 {len(all_potential_entries)} 个潜在 JSON 条目。")

    if not all_potential_entries:
        print("没有生成任何有效条目。JSON文件将为空。")
        selected_entries = []
    elif len(all_potential_entries) <= MAX_ENTRIES:
        print(f"生成的条目数 ({len(all_potential_entries)}) 未超过 MAX_ENTRIES ({MAX_ENTRIES})，全部保留 (仍按置信度排序)。")
        all_potential_entries.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
        selected_entries = all_potential_entries
    else: # len(all_potential_entries) > MAX_ENTRIES
        print(f"生成的条目数 ({len(all_potential_entries)}) 超过 MAX_ENTRIES ({MAX_ENTRIES})，将进行筛选。")
        all_potential_entries.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)

        threshold_confidence = all_potential_entries[MAX_ENTRIES - 1].get('confidence', 0.0)
        # all_potential_entries[MAX_ENTRIES] is safe because len > MAX_ENTRIES
        confidence_after_threshold = all_potential_entries[MAX_ENTRIES].get('confidence', -1.0) 

        if threshold_confidence > confidence_after_threshold:
            print(f"无边界置信度平局 (阈值 {threshold_confidence:.4f} > 下一个 {confidence_after_threshold:.4f})，直接选取置信度最高的 MAX_ENTRIES 条目。")
            selected_entries = all_potential_entries[:MAX_ENTRIES]
        else: # threshold_confidence == confidence_after_threshold (tie at boundary)
            print(f"检测到边界置信度平局 (阈值 {threshold_confidence:.4f})，将进行随机筛选。")
            
            guaranteed_selection = [
                entry for entry in all_potential_entries 
                if entry.get('confidence', 0.0) > threshold_confidence
            ]
            
            candidates_at_threshold = [
                entry for entry in all_potential_entries
                if abs(entry.get('confidence', 0.0) - threshold_confidence) < 1e-9 # Float comparison
            ]
            
            num_needed_from_candidates = MAX_ENTRIES - len(guaranteed_selection)
            
            if num_needed_from_candidates < 0: # Should not happen if logic is correct
                print(f"警告: 计算得到的 'num_needed_from_candidates' ({num_needed_from_candidates}) 为负数。取 guaranteed_selection 的前 MAX_ENTRIES。")
                selected_entries = guaranteed_selection[:MAX_ENTRIES]
            elif num_needed_from_candidates == 0: # All MAX_ENTRIES were in guaranteed_selection
                 selected_entries = guaranteed_selection
            else: # num_needed_from_candidates > 0
                if len(candidates_at_threshold) < num_needed_from_candidates:
                    print(f"警告: 阈值处的候选条目 ({len(candidates_at_threshold)}) 少于所需 ({num_needed_from_candidates})。"
                          f"将选取所有候选条目，最终条目数可能少于 MAX_ENTRIES。")
                    selected_from_candidates = candidates_at_threshold # Take all available
                else:
                    random.shuffle(candidates_at_threshold)
                    selected_from_candidates = candidates_at_threshold[:num_needed_from_candidates]
                
                selected_entries = guaranteed_selection + selected_from_candidates

    print(f"已根据置信度（并在平局时随机）选择了 {len(selected_entries)} 个条目。")

    output_json_path = os.path.join(base_path, "cholec80_info.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(selected_entries, f, indent=4, ensure_ascii=False)

    print(f"处理完成！生成的包含 {len(selected_entries)} 个条目的 JSON 文件保存在：{output_json_path}")
    if len(all_potential_entries) > len(selected_entries):
        discarded_count = len(all_potential_entries) - len(selected_entries)
        print(f"由于筛选和限制，有 {discarded_count} 个条目被丢弃。")