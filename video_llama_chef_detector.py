import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import cv2
import os
from typing import Dict, List
import json

class VideoLLaMAChefDetector:
    def __init__(self):
        print("正在初始化检测器...")
        
        # 设置文件夹
        self.video_folder = "./videos"
        self.results_folder = "./results"
        os.makedirs(self.video_folder, exist_ok=True)
        os.makedirs(self.results_folder, exist_ok=True)
        
        # 加载人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print("初始化完成！")
    
    def detect_chef_hat(self, head_region: np.ndarray) -> dict:
        """使用颜色分割和形状分析检测厨师帽"""
        try:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant specialized in detecting chef hats in images."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": {"image_path": frame_path}},
                        {"type": "text", "text": "Please analyze if there are any people wearing chef hats in this image. Respond in the format: 'Detected X people, Y wearing chef hats, Z without chef hats'"},
                    ]
                }
            ]
            
            # 处理输入
            inputs = self.processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            # 将输入移动到GPU
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            
            # 生成回答
            output_ids = self.model.generate(**inputs, max_new_tokens=128)
            response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # 解析回答
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
                if frame_count % sample_interval == 0:
                    print(f"\r处理进度: {frame_count}/{total_frames} 帧", end="")
                    
                    # 保存当前帧为临时图片
                    temp_frame_path = os.path.join(self.results_folder, f"temp_frame_{frame_count}.jpg")
                    cv2.imwrite(temp_frame_path, frame)
                    
                    # 分析当前帧
                    analysis = self.analyze_frame(temp_frame_path)
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
                    
                    # 删除临时文件
                    os.remove(temp_frame_path)
            
            cap.release()
            print("\n视频分析完成！")
            
            # 计算平均佩戴率
            total_analyzed = stats["total_with_hat"] + stats["total_without_hat"]
            if total_analyzed > 0:
                stats["average_hat_ratio"] = stats["total_with_hat"] / total_analyzed
            else:
                stats["average_hat_ratio"] = 0
            
            # 保存分析报告
            report_path = os.path.join(self.results_folder, f"videollama_analysis_{video_name}.json")
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
        # 初始化检测器
        detector = VideoLLaMAChefDetector()
        
        # 扫描视频文件夹
        video_files = [f for f in os.listdir(detector.video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
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
        stats = detector.analyze_video(video_name, sample_interval=interval)
        
        # 显示分析结果
        print("\n=== 分析结果 ===")
        summary = detector.get_summary(stats)
        print(summary)
        
        print(f"\n详细分析报告已保存到: results/videollama_analysis_{video_name}.json")
        
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 