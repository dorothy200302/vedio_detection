import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
from typing import List, Dict
import json
import time
from PIL import Image
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import math
import traceback
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from torch.utils.data import Subset, SubsetRandomSampler
from sklearn.metrics import confusion_matrix, roc_auc_score

class ChefHatDataset(Dataset):
    def __init__(self, image_dir: str, annotation_file: str, transform=None):
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.1)
        ]) if transform is None else transform
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
            
        # 打印数据集统计信息
        hat_count = sum(1 for ann in self.annotations if ann['has_hat'] == 1)
        no_hat_count = sum(1 for ann in self.annotations if ann['has_hat'] == 0)
        print(f"\n数据集统计:")
        print(f"总图片数: {len(self.annotations)}")
        print(f"戴帽子图片: {hat_count}")
        print(f"不戴帽子图片: {no_hat_count}")
        print(f"正负样本比例: {hat_count/no_hat_count:.2f}")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.image_dir, ann['image'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(ann['has_hat'], dtype=torch.float32)
        return image, label

class ChefHatDetector:
    def __init__(self):
        try:
            # 设置设备
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用设备: {self.device}")
            
            print("正在初始化模型...")
            # 完全禁用COCO数据集下载
            os.environ['YOLO_VERBOSE'] = 'False'
            self.model = YOLO("yolov8n.pt", task='detect')
            self.model.overrides['data'] = None
            self.model.overrides['resume'] = False
            self.model.overrides['pretrained'] = False
            
            # 将模型移动到正确的设备
            self.model = self.model.to(self.device)
            
            # 确保模型参数可以计算梯度
            for param in self.model.parameters():
                param.requires_grad = True
            
            # 使用更复杂的分类器
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.AdaptiveAvgPool2d((1, 1))
            ).to(self.device)
            
            self.classifier = nn.Sequential(
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(128, 1)
            ).to(self.device)
            
            print("模型初始化完成")
            
            # 设置类别名称（用于过滤检测结果）
            self.target_classes = {
                'person': 0,     # 人物
                'tie': 31,       # 领带（可能表示厨师服装）
                'bottle': 39,    # 瓶子（厨房常见物品）
                'bowl': 45,      # 碗
                'cup': 41,       # 杯子
                'knife': 49      # 刀具
            }
            
            print("类别配置完成")
            
            # 设置文件夹
            self.video_folder = "./videos"
            self.results_folder = "./results"
            self.dataset_folder = "./dataset"
            os.makedirs(self.video_folder, exist_ok=True)
            os.makedirs(self.results_folder, exist_ok=True)
            os.makedirs(self.dataset_folder, exist_ok=True)
            print("文件夹配置完成")
            
            # 设置评估指标记录器
            self.metrics = {
                'train_loss': [],
                'val_accuracy': [],
                'val_precision': [],
                'val_recall': [],
                'val_f1': [],
                'val_auc': []
            }
            
            # 添加最佳模型保存路径
            self.best_model_path = os.path.join(self.results_folder, 'best_model.pth')
            
            # 初始化最佳性能指标
            self.best_f1 = 0.0
            self.patience = 5  # 早停耐心值
            self.patience_counter = 0
            
            # 初始化权重
            self._initialize_weights()
            
            # 添加预测时的图像预处理
            self.predict_transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            print(f"初始化检测器时出错: {str(e)}")
            print("错误详情:")
            print(traceback.format_exc())
            raise
    
    def _initialize_weights(self):
        """初始化模型权重"""
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.feature_extractor.apply(init_weights)
        self.classifier.apply(init_weights)
    
    def save_model(self):
        """保存模型状态"""
        torch.save({
            'feature_extractor_state': self.feature_extractor.state_dict(),
            'classifier_state': self.classifier.state_dict(),
            'metrics': self.metrics
        }, self.best_model_path)
        print(f"\n模型已保存至: {self.best_model_path}")
    
    def load_model(self):
        """加载模型状态"""
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path)
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state'])
            self.classifier.load_state_dict(checkpoint['classifier_state'])
            self.metrics = checkpoint['metrics']
            print(f"\n已加载模型: {self.best_model_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线并保存"""
        # 创建一个2x2的子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('训练过程监控', fontsize=16)
        
        # 1. 训练损失曲线
        epochs = range(1, len(self.metrics['train_loss']) + 1)
        ax1.plot(epochs, self.metrics['train_loss'], 'b-', label='训练损失')
        ax1.set_title('训练损失曲线')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 验证准确率曲线
        ax2.plot(epochs, self.metrics['val_accuracy'], 'g-', label='验证准确率')
        ax2.set_title('验证准确率曲线')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # 3. 精确率和召回率曲线
        ax3.plot(epochs, self.metrics['val_precision'], 'r-', label='精确率')
        ax3.plot(epochs, self.metrics['val_recall'], 'b-', label='召回率')
        ax3.set_title('精确率和召回率曲线')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True)
        
        # 4. F1分数和AUC曲线
        ax4.plot(epochs, self.metrics['val_f1'], 'g-', label='F1分数')
        ax4.plot(epochs, self.metrics['val_auc'], 'y-', label='AUC')
        ax4.set_title('F1分数和AUC曲线')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.results_folder, 'training_curves.png'))
        print(f"\n训练曲线已保存至: {os.path.join(self.results_folder, 'training_curves.png')}")
        plt.close()
    
    def _smooth_labels(self, labels: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
        """手动实现标签平滑"""
        with torch.no_grad():
            labels = labels * (1 - smoothing) + smoothing / 2
        return labels

    def train(self, train_loader, val_loader, num_epochs=30, learning_rate=5e-5):
        """训练模型"""
        try:
            print("\n=== 开始训练过程 ===", flush=True)
            print(f"设备: {self.device}", flush=True)
            print(f"学习率: {learning_rate}", flush=True)
            print(f"训练轮数: {num_epochs}", flush=True)
            print(f"批次大小: {train_loader.batch_size}", flush=True)
            print(f"优化器: AdamW with AMSGrad", flush=True)
            print(f"学习率调度: OneCycleLR", flush=True)
            
            # 使用带权重的BCE损失，增加正样本权重
            pos_weight = torch.tensor([2.0]).to(self.device)  # 增加正样本权重
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
            
            # 使用AdamW优化器，调整参数
            optimizer = optim.AdamW(
                list(self.feature_extractor.parameters()) + 
                list(self.classifier.parameters()), 
                lr=learning_rate,
                weight_decay=0.01,  # 减小权重衰减
                betas=(0.9, 0.999),  # 使用默认动量参数
                eps=1e-8,
                amsgrad=True
            )
            
            # 调整One Cycle学习率调度参数
            steps_per_epoch = len(train_loader)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                epochs=num_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,    # 调整初始学习率的除数
                final_div_factor=1000.0  # 调整最终学习率的除数
            )
            
            # 添加指数移动平均
            ema = torch.optim.swa_utils.AveragedModel(self.feature_extractor)
            ema_classifier = torch.optim.swa_utils.AveragedModel(self.classifier)
            
            for epoch in range(num_epochs):
                print(f"\n--- Epoch {epoch+1}/{num_epochs} ---", flush=True)
                self.feature_extractor.train()
                self.classifier.train()
                total_loss = 0
                
                for batch_idx, (images, labels) in enumerate(train_loader):
                    # 将数据移动到正确的设备
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # 应用标签平滑
                    smoothed_labels = self._smooth_labels(labels, smoothing=0.05)  # 减小标签平滑程度
                    
                    optimizer.zero_grad()
                    
                    # 前向传播
                    features = self.feature_extractor(images)
                    features = features.view(features.size(0), -1)
                    outputs = self.classifier(features)
                    outputs = outputs.squeeze()
                    
                    # 计算损失
                    loss = criterion(outputs, smoothed_labels)
                    
                    # 反向传播
                    loss.backward()
                    
                    # 更严格的梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        list(self.feature_extractor.parameters()) + 
                        list(self.classifier.parameters()),
                        max_norm=0.5  # 减小梯度裁剪阈值
                    )
                    
                    optimizer.step()
                    scheduler.step()
                    
                    # 更新EMA模型
                    ema.update_parameters(self.feature_extractor)
                    ema_classifier.update_parameters(self.classifier)
                    
                    total_loss += loss.item()
                    
                    if batch_idx % 1 == 0:  # 每个批次都打印
                        current_lr = scheduler.get_last_lr()[0]
                        print(f'Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}', flush=True)
                
                avg_loss = total_loss / len(train_loader)
                print(f'Average Loss: {avg_loss:.4f}', flush=True)
                
                # 验证
                val_metrics = self.evaluate_model(val_loader.dataset)
                
                # 更新指标
                self.metrics['train_loss'].append(avg_loss)
                self.metrics['val_accuracy'].append(val_metrics['accuracy'])
                self.metrics['val_precision'].append(val_metrics['precision'])
                self.metrics['val_recall'].append(val_metrics['recall'])
                self.metrics['val_f1'].append(val_metrics['f1_score'])
                self.metrics['val_auc'].append(val_metrics['auc'])
                
                print("\n验证集性能指标:", flush=True)
                print(f"Accuracy: {val_metrics['accuracy']:.4f}", flush=True)
                print(f"Precision: {val_metrics['precision']:.4f}", flush=True)
                print(f"Recall: {val_metrics['recall']:.4f}", flush=True)
                print(f"F1-score: {val_metrics['f1_score']:.4f}", flush=True)
                print(f"AUC: {val_metrics['auc']:.4f}", flush=True)
                
                # 检查是否为最佳模型
                if val_metrics['f1_score'] > self.best_f1:
                    self.best_f1 = val_metrics['f1_score']
                    self.save_model()
                    print("\n发现更好的模型！已保存。", flush=True)
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    print(f"\n模型性能未提升，耐心值: {self.patience_counter}/{self.patience}", flush=True)
                
                # 早停检查
                if self.patience_counter >= self.patience:
                    print(f"\n连续{self.patience}轮性能未提升，停止训练", flush=True)
                    break
            
            print("\n=== 训练完成 ===", flush=True)
            
            # 加载最佳模型
            self.load_model()
            
            # 绘制训练曲线
            self.plot_training_curves()
            
        except Exception as e:
            print(f"\n训练时出错: {str(e)}", flush=True)
            print("错误详情:", flush=True)
            print(traceback.format_exc(), flush=True)
            raise
    
    def predict(self, image_path: str) -> Dict:
        """预测单张图片"""
        try:
            self.feature_extractor.eval()
            self.classifier.eval()
            
            # 加载并预处理图片
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.predict_transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(image_tensor)
                features = features.view(features.size(0), -1)
                outputs = self.classifier(features)
                probability = torch.sigmoid(outputs).item()
                prediction = int(probability > 0.5)
            
            return {
                'prediction': prediction,
                'probability': probability,
                'label': "戴帽子" if prediction == 1 else "未戴帽子"
            }
            
        except Exception as e:
            print(f"预测时出错: {str(e)}")
            print("错误详情:")
            print(traceback.format_exc())
            return None
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig(os.path.join(self.results_folder, 'confusion_matrix.png'))
        plt.close()
        print(f"\n混淆矩阵已保存至: {os.path.join(self.results_folder, 'confusion_matrix.png')}")
    
    def cross_validate(self, dataset, k_folds=5):
        """进行k折交叉验证"""
        print(f"\n开始{k_folds}折交叉验证...")
        
        # 初始化指标记录
        cv_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        # 创建k折分割器
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # 遍历每一折
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), 1):
            print(f"\n训练第{fold}折...")
            
            # 创建数据加载器
            train_subsampler = SubsetRandomSampler(train_idx)
            val_subsampler = SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(dataset, batch_size=16, sampler=train_subsampler)
            val_loader = DataLoader(dataset, batch_size=16, sampler=val_subsampler)
            
            # 重新初始化模型
            self._initialize_weights()
            
            # 训练模型
            self.train(train_loader, val_loader)
            
            # 评估性能
            metrics = self.evaluate_model(Subset(dataset, val_idx))
            
            # 记录指标
            cv_metrics['accuracy'].append(metrics['accuracy'])
            cv_metrics['precision'].append(metrics['precision'])
            cv_metrics['recall'].append(metrics['recall'])
            cv_metrics['f1'].append(metrics['f1_score'])
            cv_metrics['auc'].append(metrics['auc'])
        
        # 计算平均指标
        print("\n交叉验证结果:")
        for metric, values in cv_metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            print(f"{metric}: {mean_value:.4f} ± {std_value:.4f}")
    
    def evaluate_model(self, val_dataset: Dataset) -> Dict:
        """评估模型性能"""
        self.feature_extractor.eval()
        self.classifier.eval()
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        all_preds = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                features = self.feature_extractor(images)
                features = features.view(features.size(0), -1)
                outputs = self.classifier(features)
                outputs = outputs.squeeze()
                
                scores = torch.sigmoid(outputs)
                preds = (scores > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
        
        # 转换为numpy数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(all_labels, all_preds)
        
        # 计算指标
        metrics = {
            'accuracy': float(accuracy_score(all_labels, all_preds)),
            'precision': float(precision_score(all_labels, all_preds, zero_division=0)),
            'recall': float(recall_score(all_labels, all_preds, zero_division=0)),
            'f1_score': float(f1_score(all_labels, all_preds, zero_division=0)),
            'auc': float(roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0)
        }
        
        return metrics

def main():
    try:
        # 创建检测器实例
        detector = ChefHatDetector()
        
        # 加载数据集
        dataset = ChefHatDataset(
            image_dir="./dataset",
            annotation_file="./dataset/annotations.json"
        )
        
        # 进行交叉验证
        detector.cross_validate(dataset, k_folds=5)
        
        # 在完整数据集上训练最终模型
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # 训练最终模型
        detector.train(train_loader, val_loader, num_epochs=30, learning_rate=5e-5)
        
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        print("错误详情:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 