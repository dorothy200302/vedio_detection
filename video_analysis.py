import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from videollama.model import VideoLLAMA3
from videollama.processor import VideoProcessor
from videollama.utils import load_video

def initialize_model():
    """初始化VideoLLaMA3模型"""
    model = VideoLLAMA3.from_pretrained("DAMO-NLP-SG/VideoLLaMA3")
    tokenizer = AutoTokenizer.from_pretrained("DAMO-NLP-SG/VideoLLaMA3")
    processor = VideoProcessor()
    return model, tokenizer, processor

def analyze_video(video_path, question=None):
    """分析视频内容
    
    Args:
        video_path (str): 视频文件路径
        question (str, optional): 关于视频的具体问题
    
    Returns:
        str: 模型的回答
    """
    # 初始化模型
    model, tokenizer, processor = initialize_model()
    
    # 加载视频
    video = load_video(video_path)
    
    # 处理视频帧
    video_features = processor(video)
    
    # 准备提示词
    if question:
        prompt = f"Please analyze this video and answer the following question: {question}"
    else:
        prompt = "Please describe what you see in this video in detail."
    
    # 生成回答
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            video_features=video_features,
            max_length=500,
            temperature=0.7,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # 示例使用
    video_path = "path/to/your/video.mp4"  # 替换为实际的视频路径
    
    # 基本视频描述
    description = analyze_video(video_path)
    print("Video Description:")
    print(description)
    
    # 针对具体问题的分析
    question = "What are the main actions happening in this video?"
    answer = analyze_video(video_path, question)
    print("\nQuestion Analysis:")
    print(answer)

if __name__ == "__main__":
    main() 