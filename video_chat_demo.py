from transformers import AutoProcessor, AutoModelForImageTextToText
import os
import cv2
import numpy as np
from typing import List, Dict
import json
import torch

class VideoChat:
    def __init__(self):
        print("正在加载模型...")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.video_folder = "./videos"
        os.makedirs("results", exist_ok=True)
        print("模型加载完成！")
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """分析单帧图像"""
        try:
            # 将OpenCV的BGR格式转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 构建提示词
            prompt = "请仔细观察图像中的人物是否戴着厨师帽。如果有多个人，请分别说明每个人的帽子佩戴情况。回答格式：'检测到X人，其中Y人戴帽子，Z人未戴帽子'"
            
            # 处理图像
            inputs = self.processor(
                images=rgb_frame,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=3,
                    do_sample=False
                )
            
            # 解码回答
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "response": response,
                "frame_analysis": self._parse_response(response)
            }
            
        except Exception as e:
            print(f"分析帧时出错: {str(e)}")
            return {
                "response": str(e),
                "frame_analysis": {"total": 0, "with_hat": 0, "without_hat": 0}
            }
    
    def _parse_response(self, response: str) -> Dict:
        """解析模型回答，提取数字信息"""
        try:
            # 尝试从回答中提取数字
            import re
            numbers = re.findall(r'\d+', response)
            if len(numbers) >= 3:
                return {
                    "total": int(numbers[0]),
                    "with_hat": int(numbers[1]),
                    "without_hat": int(numbers[2])
                }
            return {"total": 0, "with_hat": 0, "without_hat": 0}
        except:
            return {"total": 0, "with_hat": 0, "without_hat": 0}
    
    def analyze_video(self, video_name: str, sample_interval: int = 30) -> Dict:
        """分析视频内容"""
        try:
            video_path = os.path.join(self.video_folder, video_name)
            if not os.path.exists(video_path):
                return {"error": "视频文件不存在"}
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "无法打开视频文件"}
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # 初始化统计信息
            stats = {
                "total_people": 0,
                "total_with_hat": 0,
                "total_without_hat": 0,
                "frame_analyses": []
            }
            
            frame_count = 0
            print("\n开始分析视频...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % sample_interval == 0:  # 每隔一定帧数分析一次
                    print(f"\r处理进度: {frame_count}/{total_frames} 帧", end="")
                    
                    # 分析当前帧
                    analysis = self.analyze_frame(frame)
                    frame_stats = analysis["frame_analysis"]
                    
                    # 更新统计信息
                    stats["total_people"] += frame_stats["total"]
                    stats["total_with_hat"] += frame_stats["with_hat"]
                    stats["total_without_hat"] += frame_stats["without_hat"]
                    
                    # 保存帧分析结果
                    stats["frame_analyses"].append({
                        "frame_number": frame_count,
                        "analysis": analysis["response"],
                        "stats": frame_stats
                    })
            
            cap.release()
            print("\n视频分析完成！")
            
            # 计算平均佩戴率
            total_analyzed = stats["total_with_hat"] + stats["total_without_hat"]
            if total_analyzed > 0:
                stats["average_hat_ratio"] = stats["total_with_hat"] / total_analyzed
            else:
                stats["average_hat_ratio"] = 0
            
            # 保存分析报告
            report_path = os.path.join("results", f"qwen_analysis_{video_name}.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            return stats
            
        except Exception as e:
            print(f"分析视频时出错: {str(e)}")
            return {"error": str(e)}
    
    def get_summary(self, stats: Dict) -> str:
        """生成分析摘要"""
        if "error" in stats:
            return f"分析失败: {stats['error']}"
        
        total_analyzed = stats["total_with_hat"] + stats["total_without_hat"]
        if total_analyzed == 0:
            return "未检测到任何人物"
        
        hat_ratio = stats["total_with_hat"] / total_analyzed * 100
        
        summary = f"""视频分析结果：
- 检测到的总人数：{total_analyzed}
- 戴帽子的人数：{stats['total_with_hat']}
- 未戴帽子的人数：{stats['total_without_hat']}
- 帽子佩戴率：{hat_ratio:.1f}%

评估："""
        
        if hat_ratio >= 90:
            summary += "\n✅ 大多数人都正确佩戴了帽子，符合规范要求。"
        elif hat_ratio >= 70:
            summary += "\n⚠️ 帽子佩戴情况一般，建议加强佩戴意识。"
        else:
            summary += "\n❌ 帽子佩戴率较低，需要立即改进！"
        
        return summary

def main():
    try:
        chat = VideoChat()
        
        # 扫描视频文件夹
        video_files = [f for f in os.listdir(chat.video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        if not video_files:
            print("错误：未找到视频文件")
            return
        
        # 显示可用的视频文件
        print("\n=== 可用的视频文件 ===")
        for i, video in enumerate(video_files, 1):
            print(f"{i}. {video}")
        
        # 选择视频文件
        while True:
            try:
                choice = int(input("\n请选择要分析的视频文件编号："))
                if 1 <= choice <= len(video_files):
                    break
                print("无效的选择，请重试")
            except ValueError:
                print("请输入有效的数字")
        
        video_name = video_files[choice - 1]
        
        # 设置采样间隔
        interval = int(input("\n请输入采样间隔（每隔多少帧分析一次，建议30-60）："))
        
        # 分析视频
        print(f"\n开始分析视频：{video_name}")
        stats = chat.analyze_video(video_name, sample_interval=interval)
        
        # 显示分析结果
        print("\n=== 分析结果 ===")
        summary = chat.get_summary(stats)
        print(summary)
        
        print(f"\n详细分析报告已保存到: results/qwen_analysis_{video_name}.json")
        
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 