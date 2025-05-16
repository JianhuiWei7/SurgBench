import os
import ffmpeg
from tqdm import tqdm  # 导入 tqdm
import glob
import json
from collections import Counter

def split_video(input_file, start_time, end_time, output_file):
    duration = end_time - start_time
    try:
        (
            ffmpeg
            .input(input_file, ss=start_time)
            .output(output_file, to=duration)
            .run(overwrite_output=True)
        )
        print(f"视频已成功分割并保存为 {output_file}")
        return True
    except ffmpeg.Error as e:
        print(f"发生错误: {e.stderr.decode()}")
        return False
        
def get_frame_rate(input_file):
    try:
        # 获取视频信息
        probe = ffmpeg.probe(input_file)
        
        # 提取流信息
        video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
        
        if video_streams:
            frame_rate = video_streams[0].get('r_frame_rate', 'N/A')
            print(f"帧率: {frame_rate}")
        else:
            print("未找到视频流。")
    except ffmpeg.Error as e:
        print(f"发生错误: {e.stderr.decode()}")
def split_smaller_clips(clip, segment_length=60):
    start_frame = int(clip[0])
    end_frame = int(clip[1])
    segments = []
    
    current_start = start_frame
    while current_start + segment_length <= end_frame:
        current_end = current_start + segment_length
        # 创建新段落的列表
        new_segment = [
            str(current_start),
            str(current_end),
            *clip[2:]  # 保留原始列表中的其余元素
        ]
        segments.append(new_segment)
        current_start = current_end
    
    return segments

if __name__ == "__main__":
    FPS = 30
    label2id = {
        "G1":0,
        "G2":1,
        "G3":2,
        "G4":3,
        "G5":4,
        "G6":5,
        "G8":6,
        "G9":7,
        "G10":8,
        "G11":9,
        "G12":10,
        "G13":11,
        "G14":12,
        "G15":13,
    }
    label2description = {
        "G1": "Reaching for needle with right hand",
        "G2": "Positioning needle",
        "G3": "Pushing needle through tissue",
        "G4": "Transferring needle from left to right",
        "G5": "Moving to center with needle in grip",
        "G6": "Pulling suture with left hand",
        "G7": "Pulling suture with right hand",
        "G8": "Orienting needle",
        "G9": "Using right hand to help tighten suture",
        "G10": "Loosening more suture",
        "G11": "Dropping suture at end and moving to end points",
        "G12": "Reaching for needle with left hand",
        "G13": "Making C loop around right hand",
        "G14": "Reaching for suture with right hand",
        "G15": "Pulling suture with both hands",
    }
    annotation_folders = ["/rsch/zikaixiao/VideoMAEv2/data_factory/RAW/JIGSAWS/Needle_Passing/Needle_Passing/transcriptions/*.txt",
                         "/rsch/zikaixiao/VideoMAEv2/data_factory/RAW/JIGSAWS/Suturing/Suturing/transcriptions/*.txt",
                         "/rsch/zikaixiao/VideoMAEv2/data_factory/RAW/JIGSAWS/Knot_Tying/Knot_Tying/transcriptions/*.txt"]
    all_instance_list = []
    video_list = []
    for annotation_folder in annotation_folders:
        for annotation_file in glob.glob(annotation_folder):
            with open(annotation_file, "r") as file:
                lines = file.readlines()
                for line in lines:
                    values = line.strip().split()  # 使用 strip() 去掉换行符
                    values.append(os.path.basename(annotation_file).strip('.txt'))
                    video_list.append(values)
    label_list = [item[2] for item in video_list]
    count = Counter(label_list)
    video_folders = ["/rsch/zikaixiao/VideoMAEv2/data_factory/RAW/JIGSAWS/Needle_Passing/Needle_Passing/video/",
                    "/rsch/zikaixiao/VideoMAEv2/data_factory/RAW/JIGSAWS/Knot_Tying/Knot_Tying/video/",
                    "/rsch/zikaixiao/VideoMAEv2/data_factory/RAW/JIGSAWS/Suturing/Suturing/video/"]
    for video_folder in video_folders:
        for filename in os.listdir(video_folder):
            for index, instance in enumerate(video_list):
                if instance[3] in filename:
                    video_list[index].append(video_folder+filename)
    for index,item in enumerate(video_list):
        if item[2] == 'G10':
            clips = split_smaller_clips(item)
            video_list.extend(clips)
            del video_list[index]
            
    for video_info in video_list:
        assert len(video_info) == 6
    # for video_info in video_list:
    #     video_file_path = video_info[-2]
    #     output_file_path = "video_clips/"
    #     get_frame_rate(input_file=video_file_path)
    for video_info in tqdm(video_list):
        
        start_time = round(int(video_info[0]) / FPS, 2)
        end_time = round(int(video_info[1]) / FPS, 2)
        duration = round(end_time - start_time, 2)
        video_file_paths = [video_info[-2], video_info[-1]]
        for video_file_path in video_file_paths:
            base_file_name = os.path.basename(video_file_path).strip(".avi")
            label = video_info[2]
            label_description = label2description[label].lower().replace(" ", "_")
            output_file_path = f"video_clips/JIGSAWS/{base_file_name}_{start_time}_{end_time}_{label}.mp4"
            if os.path.exists(output_file_path):
                print("File exists.")
                instance = {}
                instance['label'] = label2id[label]
                instance['label_index'] = label
                instance['label_description'] = label_description
                instance['base_path'] = "/rsch/jianhui/video_clips/"
                instance['relative_path'] = f"JIGSAWS/{base_file_name}_{start_time}_{end_time}_{label}.mp4"
                instance['duration'] = duration
                instance['dataset_name'] = "JIGSAWS"
                instance['task_type'] = "gesture_classification"
                all_instance_list.append(instance)
                continue
            if split_video(input_file=video_file_path, start_time=start_time, end_time=end_time, output_file=output_file_path):
                instance = {}
                instance['label'] = label2id[label]
                instance['label_index'] = label
                instance['label_description'] = label_description
                instance['base_path'] = "/rsch/jianhui/video_clips/"
                instance['relative_path'] = f"JIGSAWS/{base_file_name}_{start_time}_{end_time}_{label}.mp4"
                instance['duration'] = duration
                instance['dataset_name'] = "JIGSAWS"
                instance['task_type'] = "gesture_classification"
                all_instance_list.append(instance)
    save_file = "video_clips/JIGSAWS/JIGSAWS_info.json"
    with open(save_file, 'w', encoding='utf-8') as json_file:
        # all_instance是一个list，list里面的元素是字典
        json.dump(all_instance_list, json_file, ensure_ascii=False, indent=4)