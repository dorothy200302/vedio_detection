import cv2
import os
import sys
import traceback
from ultralytics import YOLO
import shutil
import torch

class VideoProcessor:
    def __init__(self):
        # 强制使用CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.device = "cpu"
        print(f"使用设备: {self.device}")
        
        # 初始化YOLO模型
        self.model = YOLO('yolov8n.pt')
        
        # 创建必要的文件夹
        self.dataset_dir = "./dataset"
        self.hat_dir = os.path.join(self.dataset_dir, "有帽")
        self.no_hat_dir = os.path.join(self.dataset_dir, "无帽")
        
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.hat_dir, exist_ok=True)
        os.makedirs(self.no_hat_dir, exist_ok=True)
    
    def process_frame(self, frame, results):
        """检查帧中是否有人"""
        for r in results:
            # 类别0是'person'
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    if box.cls.cpu().numpy()[0] == 0:  # person class
                        return True
        return False
    
    def extract_frames(self, video_path: str, sample_interval: int = 30):
        """从视频中抽取帧并处理"""
        try:
            print(f"\n处理视频：{os.path.basename(video_path)}")
            
            # 确定目标文件夹（基于文件名）
            video_name = os.path.basename(video_path)
            if "有帽" in video_name:
                target_dir = self.hat_dir
                print("分类为：有帽子")
            else:
                target_dir = self.no_hat_dir
                print("分类为：无帽子")
            
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频：{video_path}")
                return
            
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            saved_count = 0
            
            print(f"总帧数：{total_frames}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 按指定间隔处理帧
                if frame_count % sample_interval == 0:
                    try:
                        # 使用YOLO检测人物
                        results = self.model(frame, device=self.device)
                        
                        # 如果检测到人物，保存图片
                        if self.process_frame(frame, results):
                            output_path = os.path.join(
                                target_dir,
                                f"{os.path.splitext(video_name)[0]}_{frame_count:06d}.jpg"
                            )
                            cv2.imwrite(output_path, frame)
                            saved_count += 1
                            
                            print(f"\r处理进度: {frame_count}/{total_frames} 帧，已保存: {saved_count} 张", end="", flush=True)
                            
                    except Exception as e:
                        print(f"\n处理帧 {frame_count} 时出错: {str(e)}")
                        traceback.print_exc()
                        continue
            
            print(f"\n视频处理完成！共保存了 {saved_count} 张图片到 {target_dir}")
            
        except Exception as e:
            print(f"\n处理视频时出错: {str(e)}")
            traceback.print_exc()
        finally:
            if cap is not None:
                cap.release()

def main():
    try:
        # 初始化处理器
        processor = VideoProcessor()
        
        # 获取视频文件列表
        video_files = [f for f in os.listdir("./") if f.endswith(('.mp4', '.avi'))]
        
        if not video_files:
            print("未找到视频文件。")
            return
        
        print("=== 开始处理视频文件 ===")
        
        # 处理每个视频文件
        for video_file in video_files:
            try:
                video_path = os.path.join("./", video_file)
                processor.extract_frames(video_path, sample_interval=30)
            except Exception as e:
                print(f"\n处理视频 {video_file} 时出错: {str(e)}")
                continue
        
        print("\n=== 所有视频处理完成 ===")
        
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()