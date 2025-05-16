import os
import ffmpeg
from tqdm import tqdm  # 导入 tqdm
import glob
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
from collections import Counter
def split_video(input_file, start_time, end_time, output_file):
    duration = end_time-start_time
    if not os.path.exists(input_file):
        return False
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
            print("frame rate", frame_rate)
            duration = float(video_streams[0]['duration'])
            if frame_rate != 'N/A':
                numerator, denominator = map(int, frame_rate.split('/'))
                frame_rate_float = numerator / denominator
                total_frames = int(frame_rate_float * duration)
                return total_frames
            else:
                print("无法计算总帧数，因为帧率不可用。")
                return 0
        else:
            print("未找到视频流。")
    except ffmpeg.Error as e:
        print(f"发生错误: {e.stderr.decode()}")
def process_one_list_video(video_list):
    sub_instance_list = []
    frame_no = 0
    for video_info in tqdm(video_list):
        label_description = video_info[3]
        start_time = int(video_info[1])
        end_time = int(video_info[2])
        duration = round(end_time - start_time, 2)
        video_file_path = video_info[-1]
        
        base_file_name = video_info[0]
        output_file_path = f"video_clips/AVOS/{base_file_name}_{start_time}_{end_time}_{label_description}.mp4"
        if os.path.exists(output_file_path):
            print("File exists.")
            instance = {}
            instance['label'] = description2id[label_description]
            instance['label_description'] = label_description
            instance['base_path'] = "/rsch/jianhui/video_clips/"
            instance['relative_path'] = f"AVOS/{base_file_name}_{start_time}_{end_time}_{label_description}.mp4"
            instance['duration'] = duration
            instance['dataset_name'] = "AVOS"
            instance['task_type'] = "phase_classification"
            sub_instance_list.append(instance)
            frame_no += get_frame_rate(output_file_path)
            continue
        if split_video(input_file=video_file_path, start_time=start_time, end_time=end_time, output_file=output_file_path):
            instance = {}
            instance['label'] = description2id[label_description]
            instance['label_description'] = label_description
            instance['base_path'] = "/rsch/jianhui/video_clips/"
            instance['relative_path'] = f"AVOS/{base_file_name}_{start_time}_{end_time}_{label_description}.mp4"
            instance['duration'] = duration
            instance['dataset_name'] = "AVOS"
            instance['task_type'] = "phase_classification"
            sub_instance_list.append(instance)
    return sub_instance_list, frame_no
if __name__ == "__main__":
    save_file = "video_clips/AVOS/AVOS_info.json"
    annotation_file = "/rsch/zikaixiao/VideoMAEv2/data_factory/RAW/AVOS/open_surgery_temporal_annotations_Jan16.csv"
    all_instance_list = []
    video_list = []
    not_existing_video = 0
    total_frame = 0
    with open(annotation_file, "r") as file:
        content = pd.read_csv(file)
    description2id = {}
    id = 0
    for i in content.index:
        video_id = content['video_id'][i]
        start_second = content['start_seconds'][i]
        end_seconds = content['end_seconds'][i]
        label_description = content['label'][i]
        if label_description not in description2id.keys():
            description2id[label_description] = id
            id += 1
        values = [video_id, start_second, end_seconds, label_description]
        video_list.append(values)
    
    label_list = [item[-1] for item in video_list]
    count = Counter(label_list)
    video_folder = "/rsch/zikaixiao/VideoMAEv2/data_factory/RAW/AVOS/video/"
    for index, instance in enumerate(video_list):
        video_list[index].append(video_folder+instance[0]+".mp4")
    all_video_id = content['video_id'].unique()
    frame_rate = []
    for video_id in all_video_id:
        video_id = video_folder+video_id+".mp4"
        if not os.path.exists(video_id):
            not_existing_video += 1
        else:
            frame_rate.append(get_frame_rate(video_id))
    for video_info in video_list:
        assert len(video_info) == 5
    threads_number = 64
    interval = int(len(video_list) / threads_number)
    video_info_lists = [video_list[i*interval:(i+1)*interval] for i in range(threads_number)]
    sub_instance_list = [[] for _ in range(threads_number)]
    frame_no = [[] for _ in range(threads_number)]
    def process_one_list_of_videos(index):
        sub_instance_list[index], frame_no[index] = process_one_list_video(video_info_lists[index])
    with ThreadPoolExecutor(max_workers=threads_number) as executor:
        futures = [executor.submit(process_one_list_of_videos, index) for index in range(threads_number)]
        for job in as_completed(futures):
            print('Completed ', job)
    
    for sub_list in sub_instance_list:
        all_instance_list.extend(sub_list)
    for frame_no2 in frame_no:
        total_frame += frame_no2[0]
    print()
    train_set, test_set = train_test_split(all_instance_list, test_size=0.3, random_state=42)
    with open(save_file, 'w', encoding='utf-8') as json_file:
        # all_instance是一个list，list里面的元素是字典
        json.dump(all_instance_list, json_file, ensure_ascii=False, indent=4)
    with open("video_clips/AVOS/train.json", 'w', encoding='utf-8') as json_file:
        # all_instance是一个list，list里面的元素是字典
        json.dump(train_set, json_file, ensure_ascii=False, indent=4)
    with open("video_clips/AVOS/test.json", 'w', encoding='utf-8') as json_file:
        # all_instance是一个list，list里面的元素是字典
        json.dump(test_set, json_file, ensure_ascii=False, indent=4)   