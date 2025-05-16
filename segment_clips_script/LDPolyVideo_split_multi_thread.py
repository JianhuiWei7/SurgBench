import os
import ffmpeg
from tqdm import tqdm  # 导入 tqdm
import glob
import json
import random
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed


def extract_and_create_video(input_file, output_file, start_time, end_time,output_duration=10, fps=25):
    """
    从视频中指定时间段内按间隔提取帧，并生成一个新的视频。

    :param input_file: 输入视频文件路径
    :param output_file: 输出视频文件路径
    :param start_time: 开始时间（秒）
    :param end_time: 结束时间（秒）
    :param frame_interval: 帧提取间隔（秒），默认为1秒
    :param output_duration: 输出视频的时长（秒），默认为10秒
    :param fps: 输出视频的帧率，默认为25fps
    :return: 成功返回True，失败返回False
    """
    try:
        # 计算需要提取的帧数
        total_frames_output = output_duration*fps
        total_input_duration = int(end_time - start_time)
        frame_interval = total_input_duration/total_frames_output
        # total_frames = int((end_time - start_time) / frame_interval)
        # output_duration = total_frames / fps
        # 使用ffmpeg提取帧并生成新视频
        (
            ffmpeg
            .input(input_file, ss=start_time, to=end_time)
            .filter('fps', fps=1/frame_interval)  # 按间隔提取帧
            .filter('setpts', 'PTS-STARTPTS')     # 重置时间戳
            .output(output_file, t=output_duration, r=fps)  # 设置输出时长和帧率
            .run(overwrite_output=True)
        )
        print(f"视频已成功生成并保存为 {output_file}")
        return True
    except ffmpeg.Error as e:
        print(f"发生错误: {e.stderr.decode()}")
        return False
    

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
            frame_rate_str = video_streams[0].get('r_frame_rate', 'N/A')
            duration = float(probe['format']['duration'])
            if frame_rate_str != 'N/A':
                numerator, denominator = map(float, frame_rate_str.split('/'))
                frame_rate = numerator / denominator
            else:
                frame_rate = None
            return frame_rate, duration
            print(f"帧率: {frame_rate}")
        else:
            print("未找到视频流。")
    except ffmpeg.Error as e:
        print(f"发生错误: {e.stderr.decode()}")

def process_one_list_video(video_list):
    sub_instance_list = []
    for video_info in tqdm(video_list):
        if os.path.exists(video_info['save_path']):
            del video_info['save_path']
            del video_info['og_video']
            del video_info['start_time']
            sub_instance_list.append(video_info)
            continue
        if split_video(input_file=video_info['og_video'], start_time=video_info['start_time'], end_time=video_info['start_time']+video_info['duration'], output_file=video_info['save_path']):
            del video_info['save_path']
            del video_info['og_video']
            del video_info['start_time']
            sub_instance_list.append(video_info)
    return sub_instance_list

def process_one_list_video_press(video_list):
    sub_instance_list = []
    for video_info in tqdm(video_list):
        if os.path.exists(video_info['save_path']):
            del video_info['save_path']
            del video_info['og_video']
            del video_info['start_time']
            sub_instance_list.append(video_info)
            continue
        if extract_and_create_video(input_file=video_info['og_video'], start_time=video_info['start_time'], end_time=video_info['start_time']+video_info['duration'], output_file=video_info['save_path']):
            del video_info['save_path']
            del video_info['og_video']
            del video_info['start_time']
            sub_instance_list.append(video_info)
    return sub_instance_list
def split_video2(duration):
    segments = []
    current_time = 0
    
    while current_time < duration:
        # 生成一个随机的片段长度，范围在5到10秒之间
        segment_length = random.randint(5, 10)
        
        # 如果当前时间加上片段长度超过总长度，调整片段长度
        if current_time + segment_length > duration:
            segment_length = duration - current_time
        
        # 确保片段长度至少为5秒
        if segment_length >= 5:
            segments.append(current_time)
            current_time += segment_length
        else:
            duration = duration - segment_length
    durations = []
    for index, item in enumerate(segments):
        if index == len(segments)-1:
            durations.append(int(duration)-item)
        else:
            durations.append(segments[index+1] - item)
    return segments, durations


def split_video3(duration):
    number_of_clips = random.randint(2,4)
    sub_duration = round(duration / number_of_clips)
    start_times = [sub_duration*i for i in range(number_of_clips)]
    durations = [sub_duration for _ in range(number_of_clips)]

    return start_times, durations

if __name__ == "__main__":
    # DURATION = 5
    all_video_instance = []
    total_clips = 0
    video_info_list = []

    video_folders = ["/rsch/zikaixiao/VideoMAEv2/data_factory/RAW/LDPolypVideo/videos_with_polyps/*.avi"]
    for video_folder in video_folders:
        for video in glob.glob(video_folder):
            frame_rate, duration = get_frame_rate(video)
            start_times, durations = split_video3(duration)
            total_clips += len(start_times)
            base_path = os.path.basename(video)[0:-4]
            for index, start_time in enumerate(start_times):
                instance = {}
                instance['label'] = 1
                instance['label_description'] = "clips_with_polyps"
                instance['base_path'] = "/rsch/jianhui/video_clips/"
                save_path = f"video_clips/LDPolyVideo/{base_path}_{start_time}_{start_time+durations[index]}_polyps.mp4"
                instance['relative_path'] = f"LDPolyVideo/{base_path}_{start_time}_{start_time+durations[index]}_polyps.mp4"
                instance['duration'] = durations[index]
                instance['dataset_name'] = "LDPolyVideo"
                instance['task_type'] = "polyp_detection"
                instance['compress'] = "10s"
                instance['og_video'] = video
                instance['save_path'] = save_path
                instance['start_time'] = start_time
                video_info_list.append(instance)

    threads_number = 64
    interval = int(len(video_info_list) / threads_number)
    video_info_lists = [video_info_list[i*interval:(i+1)*interval] for i in range(threads_number)]
    sub_instance_list = [[] for _ in range(threads_number)]

    def process_one_list_of_videos_press(index):
        sub_instance_list[index] = process_one_list_video_press(video_info_lists[index])
    with ThreadPoolExecutor(max_workers=threads_number) as executor:
        futures = [executor.submit(process_one_list_of_videos_press, index) for index in range(threads_number)]
        for job in as_completed(futures):
            print('Completed ', job)

    for sub_list in sub_instance_list:
        all_video_instance.extend(sub_list)
    print(total_clips)
    video_info_list = []



    video_folders = ["video_clips/LDPolyVideo_remove_noise/Test/*.mp4", "video_clips/LDPolyVideo_remove_noise/TrainValid/*.mp4"]
    for video_folder in video_folders:
        for video in glob.glob(video_folder):
            frame_rate, duration = get_frame_rate(video)
            start_times,durations = split_video2(duration)
            total_clips += len(start_times)
            base_path = os.path.basename(video)[0:-4]
            for index, start_time in enumerate(start_times):
                instance = {}
                instance['label'] = 1
                instance['label_description'] = "clips_with_polyps"
                instance['base_path'] = "/rsch/jianhui/video_clips/"
                save_path = f"video_clips/LDPolyVideo/{base_path}_{start_time}_{start_time+durations[index]}_polyps.mp4"
                instance['relative_path'] = f"LDPolyVideo/{base_path}_{start_time}_{start_time+durations[index]}_polyps.mp4"
                instance['duration'] = durations[index]
                instance['dataset_name'] = "LDPolyVideo"
                instance['task_type'] = "polyp_detection"
                instance['og_video'] = video
                instance['save_path'] = save_path
                instance['start_time'] = start_time
                video_info_list.append(instance)
            # if os.path.exists(save_path):
            #     all_video_instance.append(instance)
            #     continue
            # if split_video(video, start_time, start_time+5, save_path):
            #     all_video_instance.append(instance)
    video_folders = ["/rsch/zikaixiao/VideoMAEv2/data_factory/RAW/LDPolypVideo/videos_without_polyps/*.avi", "video_clips/LDPolyVideo_remove_noise/noise_folder_no_poly/*.mp4"]
    for video_folder in video_folders:
        for video in glob.glob(video_folder):
            frame_rate, duration = get_frame_rate(video)
            start_times,durations = split_video2(duration)
            total_clips += len(start_times)
            base_path = os.path.basename(video)[0:-4]
            for index, start_time in enumerate(start_times):
                instance = {}
                instance['label'] = 0
                instance['label_description'] = "clips_without_polyps"
                instance['base_path'] = "/rsch/jianhui/video_clips/"
                save_path = f"video_clips/LDPolyVideo/{base_path}_{start_time}_{start_time+durations[index]}_no_polyps.mp4"
                instance['relative_path'] = f"LDPolyVideo/{base_path}_{start_time}_{start_time+durations[index]}_no_polyps.mp4"
                instance['duration'] = durations[index]
                instance['dataset_name'] = "LDPolyVideo"
                instance['task_type'] = "polyp_detection"
                instance['og_video'] = video
                instance['save_path'] = save_path
                instance['start_time'] = start_time
                video_info_list.append(instance)
    threads_number = 64
    interval = int(len(video_info_list) / threads_number)
    video_info_lists = [video_info_list[i*interval:(i+1)*interval] for i in range(threads_number)]
    sub_instance_list = [[] for _ in range(threads_number)]

    def process_one_list_of_videos(index):
        sub_instance_list[index] = process_one_list_video(video_info_lists[index])
    with ThreadPoolExecutor(max_workers=threads_number) as executor:
        futures = [executor.submit(process_one_list_of_videos, index) for index in range(threads_number)]
        for job in as_completed(futures):
            print('Completed ', job)

    for sub_list in sub_instance_list:
        all_video_instance.extend(sub_list)
    print(total_clips)

    train_set, test_set = train_test_split(all_video_instance, test_size=0.3, random_state=42)

    with open("video_clips/LDPolyVideo/LDPolyVideo_info.json", 'w', encoding='utf-8') as json_file:
        # all_instance是一个list，list里面的元素是字典
        json.dump(all_video_instance, json_file, ensure_ascii=False, indent=4)
    with open("video_clips/LDPolyVideo/train.json", 'w', encoding='utf-8') as json_file:
        # all_instance是一个list，list里面的元素是字典
        json.dump(train_set, json_file, ensure_ascii=False, indent=4)
    with open("video_clips/LDPolyVideo/test.json", 'w', encoding='utf-8') as json_file:
        # all_instance是一个list，list里面的元素是字典
        json.dump(test_set, json_file, ensure_ascii=False, indent=4)  