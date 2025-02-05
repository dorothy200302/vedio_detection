from openai import OpenAI
import os
from typing import List, Dict
import time

# 设置OpenAI客户端配置
client = OpenAI(
    api_key="sk-LgCHxn9whfKTRuZ0Fe0b572905Eb46C89e17Ae9186F2C231",
    base_url="https://aihubmix.com/v1",
    timeout=30.0  # 增加超时时间到30秒
)

class PromptingDemo:
    def __init__(self):
        self.client = OpenAI(
            api_key="sk-LgCHxn9whfKTRuZ0Fe0b572905Eb46C89e17Ae9186F2C231",
            base_url="https://aihubmix.com/v1",
            timeout=30.0  # 增加超时时间到30秒
        )
    
    def get_completion(self, prompt: str, max_retries: int = 3) -> str:
        """发送请求到OpenAI API并获取响应，包含重试机制"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:  # 最后一次重试
                    return f"Error: {str(e)}"
                print(f"重试 {attempt + 1}/{max_retries}...")
                time.sleep(2)  # 等待2秒后重试
    
    def zero_shot_demo(self):
        """零样本提示演示"""
        prompt = "法国的首都是什么？"
        response = self.get_completion(prompt)
        print("\n=== 零样本提示演示 ===")
        print(f"提示: {prompt}")
        print(f"响应: {response}")

    def one_shot_demo(self):
        """一次性提示演示"""
        prompt = """
        产品描述示例：
        "经典白色马克杯 - 一款永恒的12盎司陶瓷杯，是您晨间咖啡的完美之选。可微波加热，可用洗碗机清洗。"
        
        请为以下产品写一个类似的描述：游戏机械键盘
        """
        response = self.get_completion(prompt)
        print("\n=== 一次性提示演示 ===")
        print(f"提示: {prompt}")
        print(f"响应: {response}")

    def few_shot_demo(self):
        """少样本提示演示"""
        prompt = """
        将这些句子转化为过去时：
        输入："我吃一个苹果"
        输出："我吃了一个苹果"
        
        输入："她跑得快"
        输出："她跑得很快"
        
        现在转化："他们唱得好"
        """
        response = self.get_completion(prompt)
        print("\n=== 少样本提示演示 ===")
        print(f"提示: {prompt}")
        print(f"响应: {response}")

    def role_based_demo(self):
        """基于角色的提示演示"""
        prompt = "扮演网络安全专家的角色，向初学者解释密码哈希的工作原理。"
        response = self.get_completion(prompt)
        print("\n=== 基于角色的提示演示 ===")
        print(f"提示: {prompt}")
        print(f"响应: {response}")

    def prompt_reframing_demo(self):
        """提示重构演示"""
        prompt = "如何向一个10岁的孩子解释太阳能板？"
        response = self.get_completion(prompt)
        print("\n=== 提示重构演示 ===")
        print(f"提示: {prompt}")
        print(f"响应: {response}")

    def prompt_chaining_demo(self):
        """提示组合演示"""
        prompt = """
        请完成以下三个任务：
        1. 解释Python是什么
        2. 提供使用Python的三个关键好处
        3. 建议两个适合初学者的项目
        """
        response = self.get_completion(prompt)
        print("\n=== 提示组合演示 ===")
        print(f"提示: {prompt}")
        print(f"响应: {response}")

def main():
    demo = PromptingDemo()
    demo.zero_shot_demo()
    demo.one_shot_demo()
    demo.few_shot_demo()
    demo.role_based_demo()
    demo.prompt_reframing_demo()
    demo.prompt_chaining_demo()

if __name__ == "__main__":
    main() 