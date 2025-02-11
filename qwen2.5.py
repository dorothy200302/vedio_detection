import cv2
from openai import OpenAI
import os
from PIL import Image
import base64
import io

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_frame(client, frame):
    """Analyze a single frame using Qwen2.5-VL model via API"""
    # Convert frame to PIL Image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Convert image to base64
    base64_image = encode_image_to_base64(frame_pil)
    
    # Prepare query for the model
    response = client.chat.completions.create(
        model='Qwen/Qwen2.5-VL-72B-Instruct',
        messages=[{
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': '请分析这张图片中的厨师是否戴着帽子？如果出现的人都戴了帽子，请回答:是的，如果有人没有戴（即有人的头上半部分出现而没有帽子，当人的头部在画面内呈现不完全时不算），请回答:没有。',
            }, {
                'type': 'image_url',
                'image_url': {
                    'url': f"data:image/jpeg;base64,{base64_image}",
                },
            }],


        }],
        stream=False
    )
    
    return response.choices[0].message.content

def process_video(video_path, sample_interval=30):
    """Process video and analyze frames for chef hat detection"""
    print(f"开始处理视频: {video_path}")
    
    # Initialize API client
    client = OpenAI(
        base_url='https://api-inference.modelscope.cn/v1/',
        api_key='4e50e771-be7e-4a53-aef0-e8e5a982684d',
    )
    
    # Initialize counters
    hat_count = 0
    no_hat_count = 0
    total_analyzed = 0
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return
    
    frame_count = 0
    no_hat_frames = []  # Initialize list to track frames without hats
    no_hat_timestamps = []  # Initialize list to track timestamps without hats
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get video FPS
    first_frame_shape = None  # Store the shape of the first frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Store the shape of the first frame
        if first_frame_shape is None:
            first_frame_shape = frame.shape
            
        # Only process every nth frame
        if frame_count % sample_interval == 0:
            current_time = frame_count / fps  # Calculate current time in seconds
            print(f"\n分析第 {frame_count} 帧 (时间: {current_time:.2f}秒):")
            result = analyze_frame(client, frame)
            print(result)
            
            # Count hat/no hat based on result
            total_analyzed += 1
            if "是的" in result and "没有" not in result:
                hat_count += 1
            else:
                no_hat_frames.append(frame.copy())  # Make a copy of the frame
                no_hat_timestamps.append(current_time)  # Record timestamp
                no_hat_count += 1
            
        frame_count += 1
        
    cap.release()
    
    # Calculate and print statistics
    print("\n统计结果:")
    print(f"总共分析帧数: {total_analyzed}")
    print(f"戴帽子帧数: {hat_count} ({(hat_count/total_analyzed*100):.1f}%)")
    print(f"未戴帽子帧数: {no_hat_count} ({(no_hat_count/total_analyzed*100):.1f}%)")
    
    if no_hat_timestamps:
        print("\n未戴帽子的时间点:")
        for timestamp in no_hat_timestamps:
            minutes = int(timestamp // 60)
            seconds = timestamp % 60
            print(f"{minutes:02d}:{seconds:05.2f}")
    
    # Save frames without hats to a separate video
    if no_hat_frames and first_frame_shape is not None:
        output_path = video_path.replace('.mp4', '_no_hat.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, 
                            (first_frame_shape[1], first_frame_shape[0]))
        for frame in no_hat_frames:
            out.write(frame)
        out.release()
        print(f"已保存未戴帽子视频: {output_path}")
    
    print("视频处理完成")

if __name__ == "__main__":
    # Example usage
    video_path = r"C:\Users\dorot\Desktop\prompt practice\无帽13200.mp4"
    process_video(video_path)

