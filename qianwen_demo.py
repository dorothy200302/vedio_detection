import requests
import json
from typing import Dict
import urllib3

# 禁用SSL警告
urllib3.disable_warnings()

# class QianwenDemo:
#     def __init__(self):
#         self.api_key = "sk-ccbb91e15c89494a99eff2d7ffc845aa"
#         self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        
#     def get_completion(self, prompt: str) -> str:
#         """发送请求到通义千问API并获取响应"""
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
        
#         data = {
#             "model": "qwen-lite",
#             "input": {
#                 "prompt": prompt
#             }
#         }
        
#         print(f"\n正在发送请求到: {self.api_url}")
#         print(f"请求头: {headers}")
#         print(f"请求数据: {data}")
        
#         try:
#             # 禁用代理，直接连接
#             proxies = {
#                 "http": None,
#                 "https": None
#             }
#             response = requests.post(
#                 self.api_url, 
#                 headers=headers, 
#                 json=data, 
#                 timeout=60,
#                 proxies=proxies,
#                 verify=False
#             )
#             print(f"响应状态码: {response.status_code}")
#             print(f"响应内容: {response.text}")
            
#             if response.status_code == 200:
#                 return response.json()['output']['text']
#             else:
#                 return f"Error: {response.status_code} - {response.text}"
#         except Exception as e:
#             print(f"发生异常: {str(e)}")
#             return f"Error: {str(e)}"
    
#     def chat_demo(self):
#         """聊天示例"""
#         prompt = "你好，请介绍一下你自己"
#         response = self.get_completion(prompt)
#         print("\n=== 基础对话示例 ===")
#         print(f"问题: {prompt}")
#         print(f"回答: {response}")
    
#     def creative_writing(self, topic: str):
#         """创意写作示例"""
#         prompt = f"""请以'{topic}'为主题写一段富有创意的文字，要求：
#         1. 字数在200字左右
#         2. 语言优美生动
#         3. 富有想象力
#         """
#         response = self.get_completion(prompt)
#         print("\n=== 创意写作示例 ===")
#         print(f"主题: {topic}")
#         print(f"作品: {response}")
    
#     def knowledge_qa(self, question: str):
#         """知识问答示例"""
#         prompt = f"请详细回答这个问题：{question}"
#         response = self.get_completion(prompt)
#         print("\n=== 知识问答示例 ===")
#         print(f"问题: {question}")
#         print(f"回答: {response}")

# def main():
#     demo = QianwenDemo()
    
#     # 运行基础对话示例
#     demo.chat_demo()
    
#     # 运行创意写作示例
#     demo.creative_writing("未来的城市")
    
#     # 运行知识问答示例
#     demo.knowledge_qa("请解释一下量子计算的基本原理")

# if __name__ == "__main__":
#     main() 
from transformers import AutoModel, AutoTokenizer
model_path = 'Qwen2-VL'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
video_path = r"C:\Users\dorot\Desktop\prompt practice\videos\无帽.mp4"
question = "Are there any people not wearing hats in the video?"
output, chat_history = model.chat(video_path=video_path, tokenizer=tokenizer, user_prompt=question)
print(output)