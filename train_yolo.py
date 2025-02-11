from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import torch
import gc
import os

def plot_metrics(results):
    """Plot training metrics"""
    metrics = {
        'Box(P)': 'metrics/precision(B)',
        'Recall': 'metrics/recall(B)',
        'mAP50': 'metrics/mAP50(B)',
        'mAP50-95': 'metrics/mAP50-95(B)'
    }
    plt.figure(figsize=(15, 10))
    
    for i, (metric_name, metric_key) in enumerate(metrics.items(), 1):
        plt.subplot(2, 2, i)
        plt.plot(results.results_dict[metric_key], label=metric_name)
        plt.title(f'{metric_name} Curve')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def train_yolo():
    try:
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load the previously trained model if it exists
        model_path = 'runs/detect/train14/weights/best.pt'
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"Continuing training from {model_path}")
        else:
            model = YOLO('yolov8n.pt')
            print("Starting new training with pretrained YOLOv8n model")

        # Train the model with custom dataset
        results = model.train(
            data='dataset/yolo_dataset/dataset.yaml',  # path to data.yaml
            epochs=100,  # increased number of epochs
            imgsz=640,  # increased image size
            batch=16,  # increased batch size
            device='cpu',  # use CPU for training
            plots=True,  # save plots
            save=True,  # save trained model
            save_period=10,  # save checkpoint every 10 epochs
            cache=False,  # disable caching
            workers=0,  # single-thread data loading
            patience=20,  # increased early stopping patience
            optimizer='AdamW',  # use AdamW optimizer
            lr0=0.001,  # initial learning rate
            weight_decay=0.0005,  # weight decay for regularization
            project='runs/continue_training',  # new project directory
            name='continue_train',  # new name
            exist_ok=True  # overwrite existing project
        )
        
        # Plot and save metrics
        plot_metrics(results)
        
        # Print final metrics
        print("\nTraining completed. Final metrics:")
        metrics = results.results_dict
        print(f"mAP50: {metrics['metrics/mAP50(B)']:.4f}")
        print(f"mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
        print(f"Precision: {metrics['metrics/precision(B)']:.4f}")
        print(f"Recall: {metrics['metrics/recall(B)']:.4f}")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

if __name__ == '__main__':
    train_yolo() 