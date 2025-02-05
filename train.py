from ultralytics import YOLO
import os
import torch
import yaml

def train_model():
    """训练YOLOv8模型"""
    try:
        print("开始训练模型...")
        
        # 强制使用CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = "cpu"
        print(f"使用设备: {device}")
        
        # 加载预训练模型
        model = YOLO("yolov8n.pt")
        
        # 加载训练配置
        with open("training.yaml", "r") as f:
            training_config = yaml.safe_load(f)
        
        # 开始训练
        results = model.train(
            data="dataset.yaml",
            epochs=training_config["epochs"],
            imgsz=training_config["imgsz"],
            batch=training_config["batch"],
            device=device,
            verbose=True
        )
        
        print("\n训练完成！")
        print(f"模型已保存到: {model.best}")  # 最佳模型路径
        
        # 在验证集上评估模型
        print("\n在验证集上评估模型...")
        metrics = model.val()
        
        print("\n验证集评估结果:")
        print(f"准确率 (mAP50): {metrics.box.map50:.3f}")
        print(f"召回率: {metrics.box.recall:.3f}")
        print(f"精确率: {metrics.box.precision:.3f}")
        
    except Exception as e:
        print(f"训练过程出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_model()