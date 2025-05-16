import os
import subprocess
import multiprocessing
from tqdm import tqdm
import logging
import shutil
from PIL import Image

# 配置参数
INPUT_DIR = "/home/zikaixiao/zikaixiao/VideoMAEv2/data_factory/RAW/JIGSAWS"
OUTPUT_DIR = "/home/zikaixiao/zikaixiao/VideoMAEv2/data_factory/RAW_320/JIGSAWS"
CODEC_CONFIG = {
    "vcodec": "h264_nvenc",
    "cq": 22,
    "preset": "medium",
    "tune": "hq",
    "profile": "high",
    "pix_fmt": "yuv420p"
}
IMAGE_QUALITY = 85  # 图片质量 (1-100)
GPU_ID = 4
TASKS_PER_GPU = 1

# 日志配置
LOG_FILE = './video_transcode.log'
PROCESSED_LOG_FILE = './processed_files.log'

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def get_video_dimensions(input_path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'csv=p=0',
        input_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return map(int, result.stdout.strip().split(','))

def get_image_dimensions(input_path):
    with Image.open(input_path) as img:
        return img.size  # 返回 (width, height)

def generate_tasks(input_dir, output_dir):
    video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv']
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    tasks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            src = os.path.join(root, file)
            rel_path = os.path.relpath(src, input_dir)
            dest = os.path.join(output_dir, rel_path)
            ext = os.path.splitext(file)[1].lower()
            
            if ext in video_exts:
                dest = os.path.splitext(dest)[0] + '.mp4'
                tasks.append(('transcode_video', src, dest))
            elif ext in image_exts:
                tasks.append(('resize_image', src, dest))
            else:
                tasks.append(('copy', src, dest))
    return tasks

def resize_image(src, dest, target_shortest_side=320):
    try:
        with Image.open(src) as img:
            # 计算新尺寸
            width, height = img.size
            if width < height:
                new_width = target_shortest_side
                new_height = int(height * (target_shortest_side / width))
            else:
                new_height = target_shortest_side
                new_width = int(width * (target_shortest_side / height))
            
            # 调整大小并保持质量
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 创建输出目录
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            
            # 保存图片（保持原始格式）
            if src.lower().endswith('.jpg') or src.lower().endswith('.jpeg'):
                img.save(dest, quality=IMAGE_QUALITY, optimize=True)
            elif src.lower().endswith('.png'):
                img.save(dest, optimize=True)
            else:
                img.save(dest)
                
            return ("success", dest)
    except Exception as e:
        return ("failed", src, str(e))

def process_task(args):
    task_type, src, dest = args

    if os.path.exists(dest):
        return ("skipped", dest, "File exists")

    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        
        if task_type == 'transcode_video':
            width, height = get_video_dimensions(src)
            scale = "scale=320:-2" if width < height else "scale=-2:320"
            
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-hwaccel', 'cuda', '-hwaccel_device', str(GPU_ID),
                '-i', src,
                '-vf', scale,
                '-c:v', CODEC_CONFIG["vcodec"],
                '-preset', CODEC_CONFIG["preset"],
                '-tune', CODEC_CONFIG["tune"],
                '-profile:v', CODEC_CONFIG["profile"],
                '-cq', str(CODEC_CONFIG["cq"]),
                '-pix_fmt', CODEC_CONFIG["pix_fmt"],
                '-an', '-y', dest
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return ("success", dest)
            
        elif task_type == 'resize_image':
            return resize_image(src, dest)
            
        elif task_type == 'copy':
            shutil.copy2(src, dest)
            return ("success", dest)
            
    except subprocess.CalledProcessError as e:
        return ("failed", src, f"FFmpeg error: {e.stderr[:200]}")
    except Exception as e:
        return ("failed", src, str(e))

def load_processed(log_file):
    return set() if not os.path.exists(log_file) else set(open(log_file).read().splitlines())

def save_processed(log_file, path):
    with open(log_file, 'a') as f:
        f.write(f"{path}\n")

if __name__ == "__main__":
    tasks = generate_tasks(INPUT_DIR, OUTPUT_DIR)
    processed = load_processed(PROCESSED_LOG_FILE)
    
    filtered = [t for t in tasks if t[2] not in processed]
    
    with multiprocessing.Pool(TASKS_PER_GPU, initializer=init_worker) as pool:
        with tqdm(total=len(filtered), desc="Processing") as pbar:
            for result in pool.imap_unordered(process_task, filtered):
                status, *rest = result
                if status == "success":
                    save_processed(PROCESSED_LOG_FILE, rest[0])
                    logging.info(f"Success: {rest[0]}")
                elif status == "failed":
                    logging.error(f"Failed: {rest[0]} - {rest[1]}")
                pbar.update(1)

    print(f"处理完成，共处理 {len(filtered)} 个文件")