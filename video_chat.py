import cv2
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image
import os
from typing import List, Dict
import json
import time

class ChefDetector:
    def __init__(self):
        try:
            print("正在初始化YOLO模型...")
            # 下载并加载YOLO模型
            self.person_model = YOLO("yolov8n.pt")  # 使用较小的模型
            print("YOLO模型加载成功")
            
            # 设置视频文件夹
            self.video_folder = "./videos"
            if not os.path.exists(self.video_folder):
                os.makedirs(self.video_folder)
                print(f"创建视频文件夹: {self.video_folder}")
            
            # 设置结果保存文件夹
            self.results_folder = "./results"
            if not os.path.exists(self.results_folder):
                os.makedirs(self.results_folder)
                print(f"创建结果文件夹: {self.results_folder}")
                
        except Exception as e:
            print(f"初始化检测器时出错: {str(e)}")
            raise
            
    def detect_head_and_hat(self, frame: np.ndarray, person_box: List[float]) -> Dict:
        """检测头部区域是否有帽子"""
        try:
            x1, y1, x2, y2 = [int(coord) for coord in person_box]
            
            # 估算头部区域（通常在人物边界框的上部）
            head_height = int((y2 - y1) * 0.2)  # 头部大约占人物高度的20%
            head_y2 = y1 + head_height
            
            # 提取头部区域图像
            head_region = frame[y1:head_y2, x1:x2]
            
            if head_region.size == 0:
                return {"has_hat": False, "confidence": 0.0}
            
            # 转换为HSV颜色空间以更好地检测帽子
            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
            
            # 定义白色帽子的HSV范围
            white_lower = np.array([0, 0, 180])
            white_upper = np.array([180, 30, 255])
            
            # 创建掩码
            white_mask = cv2.inRange(hsv, white_lower, white_upper)
            
            # 计算白色区域的比例
            white_ratio = np.sum(white_mask > 0) / (head_region.shape[0] * head_region.shape[1])
            
            # 如果白色区域占比超过阈值，认为戴了帽子
            has_hat = white_ratio > 0.3
            
            confidence = white_ratio if has_hat else 1 - white_ratio
            
            return {
                "has_hat": has_hat,
                "confidence": float(confidence)
            }
            
        except Exception as e:
            print(f"检测帽子时出错: {str(e)}")
            return {"has_hat": False, "confidence": 0.0}
            
    def process_frame(self, frame: np.ndarray) -> Dict:
        """处理单个视频帧"""
        try:
            # 运行人物检测
            results = self.person_model(frame)
            
            # 获取检测结果
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # 获取类别
                    cls = int(box.cls[0].cpu().numpy())
                    cls_name = self.person_model.names[cls]
                    
                    # 只处理人物检测结果
                    if cls_name == "person":
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 获取置信度
                        conf = float(box.conf[0].cpu().numpy())
                        
                        # 检测是否戴帽子
                        hat_result = self.detect_head_and_hat(frame, [x1, y1, x2, y2])
                        
                        # 保存检测结果
                        detection = {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": conf,
                            "has_hat": hat_result["has_hat"],
                            "hat_confidence": hat_result["confidence"]
                        }
                        detections.append(detection)
                        
                        # 在图像上绘制边界框和标签
                        color = (0, 255, 0) if hat_result["has_hat"] else (0, 0, 255)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        status = "戴帽子" if hat_result["has_hat"] else "未戴帽子"
                        cv2.putText(frame, f"{status} ({hat_result['confidence']:.2f})", 
                                  (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return {
                "frame": frame,
                "detections": detections
            }
            
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
            return {
                "frame": frame,
                "detections": []
            }
            
    def analyze_video(self, video_name: str) -> Dict:
        """分析整个视频"""
        try:
            video_path = os.path.join(self.video_folder, video_name)
            if not os.path.exists(video_path):
                return {"error": "视频文件不存在"}
                
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "无法打开视频文件"}
                
            # 获取视频信息
            fps = float(cap.get(cv2.CAP_PROP_FPS))  # 转换为Python float
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 准备输出视频
            output_path = os.path.join(self.results_folder, f"analyzed_{video_name}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 分析结果
            frame_results = []
            hat_stats = {"with_hat": 0, "without_hat": 0}
            frame_number = 0
            
            print("\n开始分析视频...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_number += 1
                if frame_number % 5 == 0:  # 每5帧分析一次
                    print(f"\r处理进度: {frame_number}/{frame_count} 帧", end="")
                    
                    # 处理当前帧
                    result = self.process_frame(frame)
                    
                    # 更新统计信息
                    for det in result["detections"]:
                        if det["has_hat"]:
                            hat_stats["with_hat"] += 1
                        else:
                            hat_stats["without_hat"] += 1
                    
                    # 确保所有数值都是Python原生类型
                    clean_detections = []
                    for det in result["detections"]:
                        clean_det = {
                            "bbox": [float(x) for x in det["bbox"]],
                            "confidence": float(det["confidence"]),
                            "has_hat": bool(det["has_hat"]),
                            "hat_confidence": float(det["hat_confidence"])
                        }
                        clean_detections.append(clean_det)
                    
                    frame_results.append({
                        "frame_number": int(frame_number),
                        "detections": clean_detections
                    })
                
                # 写入输出视频
                out.write(frame)
            
            print("\n视频分析完成！")
            
            # 释放资源
            cap.release()
            out.release()
            
            # 生成分析报告
            report = {
                "video_info": {
                    "name": str(video_name),
                    "fps": float(fps),
                    "frame_count": int(frame_count),
                    "resolution": f"{width}x{height}"
                },
                "hat_statistics": {
                    "with_hat": int(hat_stats["with_hat"]),
                    "without_hat": int(hat_stats["without_hat"])
                },
                "frame_results": frame_results,
                "output_video": str(output_path)
            }
            
            # 保存分析报告
            report_path = os.path.join(self.results_folder, f"report_{video_name}.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            return report
            
        except Exception as e:
            print(f"分析视频时出错: {str(e)}")
            return {"error": str(e)}
            
    def get_summary(self, report: Dict) -> str:
        """生成分析摘要"""
        if "error" in report:
            return "无法生成摘要：" + report["error"]
            
        stats = report["hat_statistics"]
        total_detections = stats["with_hat"] + stats["without_hat"]
        
        if total_detections == 0:
            return "未检测到任何人物"
            
        with_hat_ratio = stats["with_hat"] / total_detections * 100
        without_hat_ratio = stats["without_hat"] / total_detections * 100
        
        summary = f"""分析结果：
- 总检测次数：{total_detections}
- 戴帽子检测次数：{stats['with_hat']} ({with_hat_ratio:.1f}%)
- 未戴帽子检测次数：{stats['without_hat']} ({without_hat_ratio:.1f}%)
        """
        
        # 添加建议
        if without_hat_ratio > 10:  # 如果超过10%的检测显示未戴帽子
            summary += "\n⚠️ 警告：检测到较多未戴帽子的情况，建议加强监管。"
            
        return summary

def main():
    try:
        print("初始化检测器...")
        detector = ChefDetector()
        
        # 获取视频文件列表
        print("\n扫描视频文件夹...")
        video_files = [f for f in os.listdir("./videos") if f.endswith(('.mp4', '.avi', '.mov'))]
        print(f"找到 {len(video_files)} 个视频文件")
        
        if not video_files:
            print("未找到视频文件。请将视频文件放在 './videos' 文件夹中。")
            return
            
        # 显示可用的视频文件
        print("\n=== 可用的视频文件 ===")
        for i, video in enumerate(video_files, 1):
            print(f"{i}. {video}")
            
        # 选择视频文件
        while True:
            try:
                choice = input("\n请选择要分析的视频文件编号：").strip()
                if not choice:  # 检查空输入
                    print("请输入一个数字。")
                    continue
                    
                choice = int(choice)
                if 1 <= choice <= len(video_files):
                    video_name = video_files[choice - 1]
                    break
                else:
                    print(f"请输入1到{len(video_files)}之间的数字。")
            except ValueError:
                print("请输入有效的数字。")
            except Exception as e:
                print(f"输入处理出错: {str(e)}")
                
        # 分析视频
        print(f"\n开始分析视频：{video_name}")
        report = detector.analyze_video(video_name)
        
        if "error" not in report:
            # 显示视频信息
            print("\n=== 视频信息 ===")
            print(f"分辨率: {report['video_info']['resolution']}")
            print(f"帧率: {report['video_info']['fps']}")
            print(f"总帧数: {report['video_info']['frame_count']}")
            
            # 显示分析结果摘要
            print("\n=== 分析结果摘要 ===")
            print(detector.get_summary(report))
            
            # 显示输出文件位置
            print("\n=== 输出文件 ===")
            print(f"分析后的视频：{report['output_video']}")
            print(f"详细报告：{os.path.join(detector.results_folder, f'report_{video_name}.json')}")
            
        else:
            print("\n视频分析失败：", report["error"])
            
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        import traceback
        print("错误详情:")
        print(traceback.format_exc())
        print("\n请检查配置和依赖是否正确安装。")

if __name__ == "__main__":
    main()