from openai import OpenAI
import os
from typing import List, Dict

class GPT4Demo:
    def __init__(self):
        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        """使用GPT-4发送请求并获取响应"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # 使用GPT-4模型
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_structured_analysis(self, text: str) -> Dict:
        """获取结构化的文本分析"""
        prompt = f"""
        请对以下文本进行分析，并返回以下方面的信息：
        1. 主要主题
        2. 关键词
        3. 情感倾向
        4. 建议的后续行动

        文本：{text}
        """
        response = self.get_completion(prompt, temperature=0.3)
        return response

    def get_creative_writing(self, topic: str) -> str:
        """生成创意写作内容"""
        prompt = f"""
        请以{topic}为主题，创作一段富有创意和想象力的文字。
        要求：
        - 使用生动的描述
        - 包含一些比喻或隐喻
        - 长度在200字左右
        """
        return self.get_completion(prompt, temperature=0.9)

    def get_code_review(self, code: str) -> str:
        """获取代码审查建议"""
        prompt = f"""
        请作为一个资深开发者，审查以下代码：
        
        {code}
        
        请提供：
        1. 代码质量评估
        2. 潜在的问题或漏洞
        3. 优化建议
        4. 最佳实践建议
        """
        return self.get_completion(prompt, temperature=0.5)

def main():
    demo = GPT4Demo()
    
    # 示例1：结构化分析
    print("\n=== 结构化分析示例 ===")
    analysis = demo.get_structured_analysis(
        "人工智能技术正在快速发展，为各行各业带来革新。"
        "但同时也带来了一些伦理和安全问题需要我们关注。"
    )
    print(analysis)
    
    # 示例2：创意写作
    print("\n=== 创意写作示例 ===")
    creative_text = demo.get_creative_writing("春天的花园")
    print(creative_text)
    
    # 示例3：代码审查
    print("\n=== 代码审查示例 ===")
    sample_code = """
    def calculate_average(numbers):
        sum = 0
        for i in numbers:
            sum += i
        return sum/len(numbers)
    """
    code_review = demo.get_code_review(sample_code)
    print(code_review)

if __name__ == "__main__":
    main() 