import cv2
import os
import sys
import traceback

def extract_frames(video_path: str, output_folder: str, sample_interval: int = 30):
    """
    从视频中抽取帧并保存为图片
    
    Args:
        video_path: 视频文件路径
        output_folder: 输出文件夹路径
        sample_interval: 采样间隔（每隔多少帧保存一次）
    """
    try:
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件：{video_path}")
            return
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 初始化计数器
        frame_count = 0
        saved_count = 0
        
        print(f"\n开始处理视频：{os.path.basename(video_path)}")
        print(f"总帧数：{total_frames}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 按指定间隔保存帧
            if frame_count % sample_interval == 0:
                try:
                    # 生成输出文件名
                    output_path = os.path.join(output_folder, f"{os.path.basename(video_path)}_{frame_count:06d}.jpg")
                    
                    # 保存图片
                    success = cv2.imwrite(output_path, frame)
                    if not success:
                        print(f"\n保存图片失败: {output_path}")
                        continue
                    
                    saved_count += 1
                    
                    # 显示进度
                    print(f"\r处理进度: {frame_count}/{total_frames} 帧，已保存: {saved_count} 张", end="", flush=True)
                except Exception as e:
                    print(f"\n保存帧 {frame_count} 时出错: {str(e)}")
                    traceback.print_exc()
                    continue
        
        print(f"\n视频处理完成！共保存了 {saved_count} 张图片到 {output_folder}")
        
    except Exception as e:
        print(f"\n处理视频时出错: {str(e)}")
        traceback.print_exc()
    finally:
        if cap is not None:
            cap.release()

def main():
    try:
        # 初始化帧提取器
        extractor = FrameExtractor()
        
        # 设置基础路径
        video_folder = "./"
        dataset_folder = "./dataset"
        
        # 确保dataset文件夹存在
        os.makedirs(dataset_folder, exist_ok=True)
        
        # 获取视频文件列表
        video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
        
        if not video_files:
            print("未找到视频文件。请将视频文件放在videos文件夹中。")
            return
        
        print("=== 开始处理视频文件 ===")
        
        # 处理每个视频文件
        for video_file in video_files:
            try:
                # 构建输入输出路径
                video_path = os.path.join(video_folder, video_file)
                output_folder = os.path.join(dataset_folder, os.path.splitext(video_file)[0])
                
                # 抽取帧
                extractor.extract_frames(video_path, output_folder, sample_interval=30)
            except Exception as e:
                print(f"\n处理视频 {video_file} 时出错: {str(e)}")
                traceback.print_exc()
                continue
        
        print("\n=== 所有视频处理完成 ===")
        
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 