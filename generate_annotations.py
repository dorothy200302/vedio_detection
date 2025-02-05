import os
import json
from typing import List, Dict

def scan_images(dataset_dir: str) -> List[Dict]:
    """扫描数据集目录下的所有图片并生成标注"""
    annotations = []
    
    # 遍历有帽和无帽两个文件夹
    hat_dir = os.path.join(dataset_dir, "有帽")
    no_hat_dir = os.path.join(dataset_dir, "无帽")
    
    # 处理有帽子的图片
    if os.path.exists(hat_dir):
        for img_name in os.listdir(hat_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                annotations.append({
                    "image": os.path.join("有帽", img_name),
                    "has_hat": 1
                })
    
    # 处理没有帽子的图片
    if os.path.exists(no_hat_dir):
        for img_name in os.listdir(no_hat_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                annotations.append({
                    "image": os.path.join("无帽", img_name),
                    "has_hat": 0
                })
    
    return annotations

def main():
    try:
        # 设置数据集目录
        dataset_dir = "./dataset"
        
        # 确保数据集目录存在
        if not os.path.exists(dataset_dir):
            print(f"错误：数据集目录 {dataset_dir} 不存在")
            return
        
        # 扫描图片并生成标注
        print("开始扫描图片...")
        annotations = scan_images(dataset_dir)
        
        if not annotations:
            print("未找到任何图片")
            return
        
        # 统计信息
        total_images = len(annotations)
        hat_images = sum(1 for ann in annotations if ann["has_hat"] == 1)
        no_hat_images = sum(1 for ann in annotations if ann["has_hat"] == 0)
        
        print(f"\n找到图片总数：{total_images}")
        print(f"戴帽子图片：{hat_images}")
        print(f"不戴帽子图片：{no_hat_images}")
        
        # 保存标注文件
        output_file = os.path.join(dataset_dir, "annotations.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        print(f"\n标注文件已保存至：{output_file}")
        
    except Exception as e:
        print(f"程序运行出错：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 