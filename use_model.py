from chef_hat_detector_fixed import ChefHatDetector
import torch
from PIL import Image
import os

def main():
    try:
        # 1. 初始化检测器
        print("初始化检测器...")
        detector = ChefHatDetector()
        
        # 2. 加载保存的模型
        print("加载模型权重...")
        detector.load_model()  # 这会加载 ./results/best_model.pth
        
        # 3. 设置为评估模式
        detector.feature_extractor.eval()
        detector.classifier.eval()
        
        # 4. 选择要预测的图片
        while True:
            image_path = input("\n请输入要预测的图片路径（输入q退出）：")
            if image_path.lower() == 'q':
                break
                
            if not os.path.exists(image_path):
                print(f"错误：文件 {image_path} 不存在")
                continue
            
            # 5. 进行预测
            result = detector.predict(image_path)
            
            if result:
                print("\n预测结果：")
                print(f"类别: {result['label']}")
                print(f"置信度: {result['probability']:.2f}")
                
                # 添加置信度解释
                if result['probability'] > 0.8:
                    print("预测非常确定")
                elif result['probability'] > 0.6:
                    print("预测比较确定")
                else:
                    print("预测不太确定，建议重新检查")
            else:
                print("预测失败")
                
    except Exception as e:
        print(f"运行出错：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()