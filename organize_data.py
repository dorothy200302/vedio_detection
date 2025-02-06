import os
import shutil
from pathlib import Path

def create_yolo_dataset_structure():
    """创建YOLO格式的数据集目录结构"""
    # 创建主目录
    directories = [
        'images/train',
        'images/val',
        'labels/train',
        'labels/val'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 复制classes.txt到主目录
    shutil.copy('classes.txt', 'data.yaml')
    
    # 创建data.yaml
    with open('data.yaml', 'w', encoding='utf-8') as f:
        f.write('''path: .
train: images/train
val: images/val

nc: 2  # number of classes
names: ['safety_hat', 'no_hat']  # class names
''')

def organize_existing_data():
    """整理现有数据到新的目录结构"""
    # 移动有帽子的图片
    hat_dir = Path('dataset/有帽')
    no_hat_dir = Path('dataset/无帽')
    
    # 创建训练集和验证集目录
    train_img_dir = Path('images/train')
    val_img_dir = Path('images/val')
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    
    # 移动图片文件
    def move_images(src_dir, train_ratio=0.8):
        files = list(src_dir.glob('*.jpg'))
        split_idx = int(len(files) * train_ratio)
        
        # 训练集
        for f in files[:split_idx]:
            shutil.copy2(f, train_img_dir / f.name)
        
        # 验证集
        for f in files[split_idx:]:
            shutil.copy2(f, val_img_dir / f.name)
    
    move_images(hat_dir)
    move_images(no_hat_dir)

def main():
    print("开始创建数据集目录结构...")
    create_yolo_dataset_structure()
    
    print("开始整理现有数据...")
    organize_existing_data()
    
    print("""
数据准备完成！接下来：
1. 运行 labelImg
2. 打开 images/train 或 images/val 目录
3. 设置标注格式为YOLO
4. 开始标注
5. 确保标注文件保存在对应的 labels 目录中
""")

if __name__ == '__main__':
    main() 