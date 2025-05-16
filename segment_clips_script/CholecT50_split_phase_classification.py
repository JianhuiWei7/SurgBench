import os
import ffmpeg
from tqdm import tqdm  # 导入 tqdm
import glob
import json
import pandas as pd
import cv2
import os
from natsort import natsorted
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
# def split_video(input_file, start_time, end_time, output_file):
#     if not os.path.exists(input_file):
#         return False
#     try:
#         (
#             ffmpeg
#             .input(input_file, ss=start_time)
#             .output(output_file, to=end_time)
#             .run(overwrite_output=True)
#         )
#         print(f"视频已成功分割并保存为 {output_file}")
#         return True
#     except ffmpeg.Error as e:
#         print(f"发生错误: {e.stderr.decode()}")
#         return False
        
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
# def process_one_list_video(video_list):
#     sub_instance_list = []
#     for video_info in tqdm(video_list):
#         label_description = video_info[3]
#         start_time = int(video_info[1])
#         end_time = int(video_info[2])
#         duration = round(end_time - start_time, 2)
#         video_file_path = video_info[-1]
        
#         base_file_name = video_info[0]
#         output_file_path = f"video_clips/AVOS/{base_file_name}_{start_time}_{end_time}_{label_description}.mp4"
#         if os.path.exists(output_file_path):
#             print("File exists.")
#             instance = {}
#             instance['label'] = description2id[label_description]
#             instance['label_description'] = label_description
#             instance['base_path'] = "/rsch/jianhui/video_clips/"
#             instance['relative_path'] = f"AVOS/{base_file_name}_{start_time}_{end_time}_{label_description}.mp4"
#             instance['duration'] = duration
#             instance['dataset_name'] = "AVOS"
#             instance['task_type'] = "phase_classification"
#             sub_instance_list.append(instance)
#             continue
#         if split_video(input_file=video_file_path, start_time=start_time, end_time=end_time, output_file=output_file_path):
#             instance = {}
#             instance['label'] = description2id[label_description]
#             instance['label_description'] = label_description
#             instance['base_path'] = "/rsch/jianhui/video_clips/"
#             instance['relative_path'] = f"AVOS/{base_file_name}_{start_time}_{end_time}_{label_description}.mp4"
#             instance['duration'] = duration
#             instance['dataset_name'] = "AVOS"
#             instance['task_type'] = "phase_classification"
#             sub_instance_list.append(instance)
#         else:
#             instance = {}
#             instance['file_exist'] = False
#             instance['label'] = description2id[label_description]
#             instance['label_description'] = label_description
#             instance['base_path'] = "/rsch/jianhui/video_clips/"
#             instance['relative_path'] = f"AVOS/{base_file_name}_{start_time}_{end_time}_{label_description}.mp4"
#             instance['duration'] = duration
#             instance['dataset_name'] = "AVOS"
#             instance['task_type'] = "phase_classification"
#             sub_instance_list.append(instance)
#     return sub_instance_list
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

def write_full_video(input_folder, output_video):
    # 获取所有 PNG 文件并按名称排序
    png_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
    png_files = natsorted(png_files)  # 按文件名自然排序（例如 1.png, 2.png, ..., 10.png）

    # 读取第一张图片，获取视频分辨率
    first_image = cv2.imread(os.path.join(input_folder, png_files[0]))
    height, width, _ = first_image.shape

    # 设置视频编码器和帧率（默认 24 FPS）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 编码器
    video_writer = cv2.VideoWriter(output_video, fourcc, 10, (width, height))

    # 逐帧写入视频
    for png_file in png_files:
        img_path = os.path.join(input_folder, png_file)
        frame = cv2.imread(img_path)
        video_writer.write(frame)

    # 释放资源
    video_writer.release()
    print("视频生成完成！")

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
if __name__ == "__main__":
    save_file = "video_clips/CholecT50/CholecT50_info.json"
    annotation_folder = "/home/zikaixiao/zikaixiao/VideoMAEv2/data_factory/RAW/CholecT50/labels/"
    video_folder = "video_clips/CholecT50_video/"
    save_video_folder = "video_clips/"
    id2phase = {
            5: "cleaning-and-coagulation",
            3: "gallbladder-dissection",
            4: "gallbladder-packaging",
            2: "clipping-and-cutting",
            0: "preparation",
            1: "carlot-triangle-dissection",
            6: "gallbladder-extraction"
        }
    total_clips_num = 0
    all_instance_list = []
    for filename in tqdm(os.listdir(annotation_folder)):
        clip_list = []
        with open(annotation_folder + filename, "r") as file:
            content = json.load(file)
        video_id = content['video']
        if video_id < 10:
            video_id = "0" + str(video_id)
        else:
            video_id = str(video_id)
        
        annotation_dict = content['annotations']
        annotation_dict = dict(sorted(annotation_dict.items(), key=lambda item: int(item[0])))
        value_now_phase = annotation_dict['0'][0][-1]
        start_frame = 0
        value_now_verb = annotation_dict['0'][0][7]
        value_now_instrument = annotation_dict['0'][0][1]
        value_now_target = annotation_dict['0'][0][8]
        for key,values in annotation_dict.items():
            value = values[0]
            if value[-1] != value_now_phase or int(key) == len(annotation_dict)-1:
                clip = {}
                clip['phase_label'] = value_now_phase
                clip['start_frame'] = start_frame
                clip['end_frame'] = int(key)-1
                start_frame = int(key)
                clip_list.append(clip)
                value_now_phase = value[-1]
            else:
                continue
        interval = 50
        clip_list_divide_by_50_frames = []
        for clip in clip_list:
            phase_label = clip['phase_label']
            start_frame = clip['start_frame']
            end_frame = clip['end_frame']
            new_clips = []
            for i in range((end_frame-start_frame) // interval):
                new_clip = {}
                new_clip['phase_label'] = phase_label
                new_clip['start_frame'] = start_frame + i*50
                new_clip['end_frame'] = start_frame + (i+1)*50
                new_clips.append(new_clip)
            clip_list_divide_by_50_frames.extend(new_clips)
        total_clips_num += len(clip_list_divide_by_50_frames)


        
        video_path = video_folder+video_id+".mp4"
        for clip in clip_list_divide_by_50_frames:
            start_time = round(int(clip['start_frame']) / 10, 2)
            end_time = round(int(clip['end_frame']) / 10, 2)
            duration = round(end_time - start_time, 2)
            label_description = id2phase[clip['phase_label']]
            relative_path = f"CholecT50_phase/{video_id}_{start_time}_{end_time}_{label_description}.mp4"
            save_path = "video_clips/" + relative_path
            if os.path.exists(save_path):
                instance = {}
                instance['label'] = clip['phase_label']
                instance['label_description'] = label_description
                instance['base_path'] = "/rsch/jianhui/video_clips/"
                instance['relative_path'] = relative_path
                instance['duration'] = duration
                instance['dataset_name'] = "CholecT50"
                instance['task_type'] = "phase_classification"
                all_instance_list.append(instance)
                continue
            if split_video(input_file=video_path, start_time=start_time, end_time=end_time, output_file=save_path):
                instance = {}
                instance['label'] = clip['phase_label']
                instance['label_description'] = label_description
                instance['base_path'] = "/rsch/jianhui/video_clips/"
                instance['relative_path'] = relative_path
                instance['duration'] = duration
                instance['dataset_name'] = "CholecT50"
                instance['task_type'] = "phase_classification"
                all_instance_list.append(instance)


    train_set, test_set = train_test_split(all_instance_list, test_size=0.3, random_state=42)

    with open("video_clips/CholecT50_phase/CholecT50_info.json", 'w', encoding='utf-8') as json_file:
        # all_instance是一个list，list里面的元素是字典
        json.dump(all_instance_list, json_file, ensure_ascii=False, indent=4)
    with open("video_clips/CholecT50_phase/train.json", 'w', encoding='utf-8') as json_file:
        # all_instance是一个list，list里面的元素是字典
        json.dump(train_set, json_file, ensure_ascii=False, indent=4)
    with open("video_clips/CholecT50_phase/test.json", 'w', encoding='utf-8') as json_file:
        # all_instance是一个list，list里面的元素是字典
        json.dump(test_set, json_file, ensure_ascii=False, indent=4)   
     



    print(total_clips_num)
    print()
        # video_subfolder = video_folder + "VID"+ str(video_id)
        # 先将图片生成视频
        # out_put_video_full = f"video_clips/CholecT50_video/{str(video_id)}.mp4"
        # frame_number += len(annotation_dict)
        # if os.path.exists(out_put_video_full):
        #     continue
        # write_full_video(video_subfolder, out_put_video_full)
        # 获取文件夹中所有文件的路径
        # file_list = [os.path.join(video_subfolder, f) for f in os.listdir(video_subfolder) if os.path.isfile(os.path.join(video_subfolder, f))]

