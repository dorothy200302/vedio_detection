import os
import anthropic
from PIL import Image
import base64
from io import BytesIO
import json
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化Claude客户端
client = anthropic.Anthropic(
    api_key=os.getenv('ANTHROPIC_API_KEY')
)

def encode_image(image_path):
    """将图片转换为base64编码"""
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

def detect_safety_hat(image_path):
    """使用Claude Vision检测安全帽"""
    # 编码图片
    base64_image = encode_image(image_path)
    
    # 构建消息
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please analyze this image and detect if there are any safety helmets/hard hats. If found, provide the following information in JSON format:\n1. Whether safety helmet is present (true/false)\n2. Location of the helmet in the image (approximate coordinates)\n3. Confidence level of detection\n4. Any notable observations about the helmet's condition or wearing status"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                }
            ]
        }]
    )
    
    try:
        # 解析响应
        response = message.content[0].text
        # 尝试提取JSON部分
        json_str = response[response.find('{'):response.rfind('}')+1]
        result = json.loads(json_str)
        return result
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {"error": str(e), "raw_response": response}

def process_directory(directory_path):
    """处理目录中的所有图片"""
    results = {}
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            print(f"Processing {filename}...")
            results[filename] = detect_safety_hat(image_path)
            
            # 保存中间结果
            with open('detection_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

def main():
    # 确保设置了API密钥
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("请设置ANTHROPIC_API_KEY环境变量")
        return
    
    # 处理训练集图片
    print("开始处理图片...")
    results = process_directory('images/train')
    
    # 保存结果
    with open('detection_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("处理完成，结果已保存到 detection_results.json")

if __name__ == "__main__":
    main() 