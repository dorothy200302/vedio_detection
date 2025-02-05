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

class ChefHatDataset(Dataset):
    def __init__(self, image_dir: str, annotation_file: str, transform=None):
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
            
        return image, torch.tensor(ann['has_hat'], dtype=torch.float32)

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = 1.0
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        return self.scale * (x @ self.lora_A.T @ self.lora_B.T)

class ChefHatDetector:
    def __init__(self):
        try:
            print("正在初始化模型...")
            # 完全禁用COCO数据集下载
            os.environ['YOLO_VERBOSE'] = 'False'
            self.model = YOLO("yolov8n.pt", task='detect')
            self.model.overrides['data'] = None
            self.model.overrides['resume'] = False
            self.model.overrides['pretrained'] = False
            
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
            )
            
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
            )
            
            # 初始化LoRA层
            self.lora_layers = {}
            self.add_lora_layers()
            
            print("YOLO模型加载成功")
            
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
            
        except Exception as e:
            print(f"初始化检测器时出错: {str(e)}")
            import traceback
            print("错误详情:")
            print(traceback.format_exc())
            raise
            
    def add_lora_layers(self):
        """为模型添加LoRA层"""
        # 为YOLO模型的关键层添加LoRA
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                in_features = module.in_channels
                out_features = module.out_channels
                self.lora_layers[name] = LoRALayer(in_features, out_features)
    
    def forward(self, x):
        """前向传播"""
        try:
            # 使用卷积层提取特征
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)
            
            # 通过分类器
            logits = self.classifier(features)
            return logits.squeeze()
            
        except Exception as e:
            print(f"前向传播时出错: {str(e)}")
            raise
    
    def evaluate_model(self, val_dataset: ChefHatDataset) -> Dict:
        """评估模型性能"""
        self.model.eval()
        self.classifier.eval()
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        all_preds = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = self.forward(images)
                scores = torch.sigmoid(outputs)
                preds = (scores > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
        
        # 转换为numpy数组以便计算
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        
        # 确保有足够的样本和两个类别的样本
        if len(np.unique(all_labels)) < 2:
            print("警告：验证集中只包含一个类别，无法计算某些指标")
            accuracy = accuracy_score(all_labels, all_preds)
            return {
                'accuracy': float(accuracy),
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc': 0.0
            }
        
        # 计算各种指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        # 计算ROC曲线和AUC
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_scores)
            roc_auc = auc(fpr, tpr)
        except ValueError as e:
            print(f"警告：计算ROC曲线时出错: {str(e)}")
            roc_auc = 0.0
        
        # 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('results/roc_curve.png')
        plt.close()
        
        # 保存评估结果
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(roc_auc)
        }
        
        with open('results/evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def train_lora(self, train_dataset: ChefHatDataset, epochs=30, batch_size=8, learning_rate=2e-4, val_split=0.2):
        """训练LoRA层并进行评估"""
        try:
            print("开始LoRA训练...")
            
            # 分割训练集和验证集
            val_size = int(len(train_dataset) * val_split)
            train_size = len(train_dataset) - val_size
            train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
            
            # 使用weighted sampler来处理类别不平衡
            labels = [train_dataset[i][1].item() for i in train_subset.indices]
            class_counts = np.bincount(labels)
            class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
            weights = class_weights[labels]
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
            
            train_loader = DataLoader(
                train_subset, 
                batch_size=batch_size, 
                sampler=sampler,
                num_workers=4,
                pin_memory=True
            )
            
            # 训练所有参数
            parameters = (
                list(self.feature_extractor.parameters()) + 
                list(self.classifier.parameters()) +
                [param for lora_layer in self.lora_layers.values() for param in [lora_layer.lora_A, lora_layer.lora_B]]
            )
            
            # 使用AdamW优化器和余弦退火学习率调度器
            optimizer = optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                anneal_strategy='cos'
            )
            
            # 使用带权重的BCE损失
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights[1]/class_weights[0]]))
            
            best_f1 = 0.0
            patience = 5  # 早停的耐心值
            no_improve = 0  # 没有改善的轮数
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.feature_extractor.to(device)
            self.classifier.to(device)
            
            print(f"\n训练集大小: {len(train_subset)}")
            print(f"验证集大小: {len(val_subset)}")
            print(f"类别分布: 无帽子 - {class_counts[0]}, 有帽子 - {class_counts[1]}")
            print(f"使用设备: {device}")
            
            # 训练循环
            for epoch in range(epochs):
                # 训练阶段
                self.model.eval()
                self.feature_extractor.train()
                self.classifier.train()
                
                total_loss = 0
                batch_count = 0
                
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = self.forward(images)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    if batch_idx % 5 == 0:
                        print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
                
                avg_loss = total_loss / batch_count
                self.metrics['train_loss'].append(avg_loss)
                
                # 验证阶段
                print("\n开始验证...")
                metrics = self.evaluate_model(val_subset)
                
                self.metrics['val_accuracy'].append(metrics['accuracy'])
                self.metrics['val_precision'].append(metrics['precision'])
                self.metrics['val_recall'].append(metrics['recall'])
                self.metrics['val_f1'].append(metrics['f1_score'])
                self.metrics['val_auc'].append(metrics['auc'])
                
                print(f"\nEpoch {epoch+1}/{epochs} 评估结果:")
                print(f"Loss: {avg_loss:.4f}")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1-score: {metrics['f1_score']:.4f}")
                print(f"AUC: {metrics['auc']:.4f}")
                
                # 检查是否需要保存最佳模型
                if metrics['f1_score'] > best_f1:
                    best_f1 = metrics['f1_score']
                    self.save_lora_weights("./best_lora_weights.pth")
                    print("保存新的最佳模型")
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"\n{patience}轮未见改善，提前停止训练")
                        break
                
                # 每轮都保存训练曲线
                self.plot_training_metrics()
            
            print("\nLoRA训练完成！")
            print(f"最佳F1分数: {best_f1:.4f}")
            print("最佳模型权重已保存到 best_lora_weights.pth")
            
        except Exception as e:
            print(f"训练LoRA时出错: {str(e)}")
            raise
    
    def plot_training_metrics(self):
        """绘制训练过程中的指标变化"""
        plt.figure(figsize=(15, 10))
        
        # 绘制损失
        plt.subplot(2, 1, 1)
        plt.plot(self.metrics['train_loss'], label='Training Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制评估指标
        plt.subplot(2, 1, 2)
        plt.plot(self.metrics['val_accuracy'], label='Accuracy')
        plt.plot(self.metrics['val_precision'], label='Precision')
        plt.plot(self.metrics['val_recall'], label='Recall')
        plt.plot(self.metrics['val_f1'], label='F1-score')
        plt.plot(self.metrics['val_auc'], label='AUC')
        plt.title('Validation Metrics over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/training_metrics.png')
        plt.close()
    
    def save_lora_weights(self, path="./lora_weights.pth"):
        """保存LoRA层的权重"""
        weights = {}
        for name, layer in self.lora_layers.items():
            weights[name] = {
                'lora_A': layer.lora_A.data,
                'lora_B': layer.lora_B.data,
                'scale': layer.scale
            }
        torch.save(weights, path)
        print(f"LoRA权重已保存到: {path}")
    
    def load_lora_weights(self, path="./lora_weights.pth"):
        """加载LoRA层的权重"""
        if os.path.exists(path):
            weights = torch.load(path)
            for name, layer_weights in weights.items():
                if name in self.lora_layers:
                    self.lora_layers[name].lora_A.data = layer_weights['lora_A']
                    self.lora_layers[name].lora_B.data = layer_weights['lora_B']
                    self.lora_layers[name].scale = layer_weights['scale']
            print("LoRA权重加载成功！")
        else:
            print("未找到LoRA权重文件")

    def detect_chef_hat(self, frame: np.ndarray, person_box: List[float], conf_threshold: float = 0.3) -> Dict:
        """使用更复杂的方法检测厨师帽和人头"""
        try:
            x1, y1, x2, y2 = [int(coord) for coord in person_box]
            
            # 扩大头部检测区域（上部30%的区域）
            head_height = int((y2 - y1) * 0.3)
            head_y1 = max(0, y1 - head_height//2)
            head_y2 = y1 + head_height
            head_x1 = max(0, x1 - 20)
            head_x2 = min(frame.shape[1], x2 + 20)
            
            # 提取头部区域
            head_region = frame[head_y1:head_y2, head_x1:head_x2]
            if head_region.size == 0:
                return {"has_head": False, "has_hat": False, "confidence": 0.0, "type": "unknown"}
            
            # 使用肤色检测判断是否有人头
            # 转换到YCrCb空间进行肤色检测
            ycrcb = cv2.cvtColor(head_region, cv2.COLOR_BGR2YCrCb)
            # 肤色范围
            min_YCrCb = np.array([0, 133, 77], np.uint8)
            max_YCrCb = np.array([235, 173, 127], np.uint8)
            
            # 创建肤色蒙版
            skin_mask = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)
            
            # 计算肤色区域比例
            skin_ratio = np.sum(skin_mask > 0) / (head_region.shape[0] * head_region.shape[1])
            has_head = skin_ratio > 0.15  # 如果肤色区域超过15%，认为检测到人头
            
            if not has_head:
                return {"has_head": False, "has_hat": False, "confidence": 0.0, "type": "unknown"}
            
            # 转换到HSV空间检测帽子
            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
            
            # 定义不同颜色的帽子范围
            color_ranges = {
                "white": ([0, 0, 180], [180, 30, 255]),
                "black": ([0, 0, 0], [180, 255, 50]),
                "blue": ([100, 50, 50], [130, 255, 255]),
            }
            
            best_confidence = 0.0
            best_color = "unknown"
            has_hat = False
            
            # 检查每种颜色
            for color_name, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                ratio = np.sum(mask > 0) / (head_region.shape[0] * head_region.shape[1])
                
                if ratio > best_confidence:
                    best_confidence = ratio
                    best_color = color_name
            
            has_hat = best_confidence > conf_threshold
            
            return {
                "has_head": True,
                "has_hat": has_hat,
                "confidence": float(best_confidence),
                "type": best_color if has_hat else "none"
            }
            
        except Exception as e:
            print(f"检测帽子时出错: {str(e)}")
            return {"has_head": False, "has_hat": False, "confidence": 0.0, "type": "error"}

    def process_frame(self, frame: np.ndarray) -> Dict:
        """处理单个视频帧"""
        try:
            results = self.model(frame, conf=0.25)
            
            detections = []
            total_heads = 0
            heads_with_hat = 0
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0].cpu().numpy())
                    cls_name = self.model.names[cls]
                    conf = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    if cls_name == "person":
                        # 检测帽子和人头
                        hat_result = self.detect_chef_hat(frame, [x1, y1, x2, y2])
                        
                        # 只有检测到人头时才计入统计
                        if hat_result["has_head"]:
                            total_heads += 1
                            if hat_result["has_hat"]:
                                heads_with_hat += 1
                        
                        detection = {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": conf,
                            "has_head": hat_result["has_head"],
                            "has_hat": hat_result["has_hat"],
                            "hat_confidence": hat_result["confidence"],
                            "hat_type": hat_result["type"]
                        }
                        detections.append(detection)
                        
                        # 在图像上绘制标注
                        if hat_result["has_head"]:
                            color = (0, 255, 0) if hat_result["has_hat"] else (0, 0, 255)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            
                            status = f"{'戴帽子' if hat_result['has_hat'] else '未戴帽子'} ({hat_result['confidence']:.2f})"
                            if hat_result["has_hat"]:
                                status += f" - {hat_result['type']}"
                                
                            cv2.putText(frame, status, 
                                      (int(x1), int(y1) - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 计算戴帽子的比例（只考虑检测到人头的情况）
            hat_ratio = heads_with_hat / total_heads if total_heads > 0 else 0
            
            # 在图像上方显示统计信息
            stats_text = f"检测到的人头数: {total_heads} | 戴帽子: {heads_with_hat} ({hat_ratio:.1%})"
            cv2.putText(frame, stats_text, 
                      (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            return {
                "frame": frame,
                "detections": detections,
                "statistics": {
                    "total_heads": total_heads,
                    "heads_with_hat": heads_with_hat,
                    "hat_ratio": float(hat_ratio)
                }
            }
            
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
            return {
                "frame": frame,
                "detections": [],
                "statistics": {
                    "total_heads": 0,
                    "heads_with_hat": 0,
                    "hat_ratio": 0.0
                }
            }

    def analyze_video(self, video_name: str) -> Dict:
        """分析整个视频"""
        try:
            video_path = os.path.join(self.video_folder, video_name)
            if not os.path.exists(video_path):
                return {"error": "视频文件不存在"}
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "无法打开视频文件"}
            
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            output_path = os.path.join(self.results_folder, f"analyzed_{video_name}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_results = []
            total_stats = {
                "total_frames_with_head": 0,
                "total_heads_detected": 0,
                "total_heads_with_hat": 0
            }
            
            frame_number = 0
            print("\n开始分析视频...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                if frame_number % 5 == 0:  # 每5帧分析一次
                    print(f"\r处理进度: {frame_number}/{frame_count} 帧", end="")
                    
                    result = self.process_frame(frame)
                    
                    # 更新统计信息
                    if result["statistics"]["total_heads"] > 0:
                        total_stats["total_frames_with_head"] += 1
                        total_stats["total_heads_detected"] += result["statistics"]["total_heads"]
                        total_stats["total_heads_with_hat"] += result["statistics"]["heads_with_hat"]
                    
                    frame_results.append({
                        "frame_number": frame_number,
                        "statistics": result["statistics"]
                    })
                
                out.write(frame)
            
            print("\n视频分析完成！")
            
            cap.release()
            out.release()
            
            # 计算总体统计信息（只考虑检测到人头的情况）
            avg_hat_ratio = (total_stats["total_heads_with_hat"] / total_stats["total_heads_detected"] 
                           if total_stats["total_heads_detected"] > 0 else 0)
            
            # 生成报告
            report = {
                "video_info": {
                    "name": str(video_name),
                    "fps": float(fps),
                    "frame_count": int(frame_count),
                    "resolution": f"{width}x{height}"
                },
                "statistics": {
                    "total_frames_analyzed": len(frame_results),
                    "frames_with_head": total_stats["total_frames_with_head"],
                    "total_heads_detected": total_stats["total_heads_detected"],
                    "total_heads_with_hat": total_stats["total_heads_with_hat"],
                    "average_hat_ratio": float(avg_hat_ratio)
                },
                "frame_results": frame_results,
                "output_video": str(output_path)
            }
            
            # 保存报告
            report_path = os.path.join(self.results_folder, f"report_{video_name}.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            return report
            
        except Exception as e:
            print(f"分析视频时出错: {str(e)}")
            return {"error": str(e)}

    def get_summary(self, report: Dict) -> str:
        """生成分析摘要"""
        if "error" in report:
            return "无法生成摘要：" + report["error"]
        
        stats = report["statistics"]
        
        if stats["total_heads_detected"] == 0:
            return "未检测到任何人头"
        
        hat_ratio = stats["total_heads_with_hat"] / stats["total_heads_detected"] * 100
        
        summary = f"""分析结果：
- 总检测帧数：{stats['total_frames_analyzed']}
- 包含人头的帧数：{stats['frames_with_head']}
- 检测到的总人头数：{stats['total_heads_detected']}
- 戴帽子的人头数：{stats['total_heads_with_hat']} ({hat_ratio:.1f}%)
- 平均戴帽率：{stats['average_hat_ratio']:.1%}
"""
        
        if hat_ratio < 90:
            summary += "\n⚠️ 警告：检测到较多未戴帽子的情况，建议加强监管。"
        
        return summary

    def process_video(self, video_path):
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 初始化结果
            results = {
                'video_info': {
                    'total_frames': total_frames,
                    'fps': fps,
                    'resolution': f"{width}x{height}"
                },
                'detections': [],
                'summary': {
                    'total_persons': 0,
                    'persons_with_hat': 0,
                    'persons_without_hat': 0,
                    'kitchen_objects': set()
                }
            }
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % 5 == 0:  # 每5帧处理一次
                    print(f"\n处理进度: {frame_count}/{total_frames} 帧")
                    
                    # 运行检测
                    detection = self.model(frame)
                    
                    # 处理检测结果
                    frame_results = []
                    for det in detection[0].boxes.data:
                        x1, y1, x2, y2, conf, cls = det
                        label = self.model.names[int(cls)]
                        frame_results.append({
                            'label': label,
                            'confidence': float(conf),  # 转换为Python float
                            'bbox': [float(x) for x in [x1, y1, x2, y2]]  # 转换为Python float列表
                        })
                        
                        # 更新统计信息
                        if label == 'person':
                            results['summary']['total_persons'] += 1
                        elif label in self.target_classes:
                            results['summary']['kitchen_objects'].add(label)
                    
                    results['detections'].append({
                        'frame_number': frame_count,
                        'objects': frame_results
                    })
            
            # 转换set为list以便JSON序列化
            results['summary']['kitchen_objects'] = list(results['summary']['kitchen_objects'])
            
            # 保存结果到JSON文件
            output_path = os.path.join('results', os.path.splitext(os.path.basename(video_path))[0] + '_analysis.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print("\n视频分析完成！")
            return results
            
        except Exception as e:
            print(f"分析视频时出错: {str(e)}")
            return None
        finally:
            if 'cap' in locals():
                cap.release()

def main():
    try:
        # 初始化检测器
        print("初始化检测器...")
        detector = ChefHatDetector()
        
        # 选择模式
        print("\n=== 请选择模式 ===")
        print("1. 训练模式")
        print("2. 视频分析模式")
        
        while True:
            try:
                mode = int(input("\n请选择模式 (1 或 2)："))
                if mode in [1, 2]:
                    break
                print("无效的选择，请重试")
            except ValueError:
                print("请输入有效的数字")
        
        if mode == 1:
            # 训练模式
            print("\n=== 开始训练模式 ===")
            
            # 创建数据集
            dataset = ChefHatDataset(
                image_dir="./dataset/images",
                annotation_file="./dataset/annotations.json"
            )
            
            print(f"加载数据集成功，共有 {len(dataset)} 张图片")
            
            # 设置训练参数
            epochs = int(input("\n请输入训练轮数 (建议20-30轮)："))
            batch_size = 4  # 减小batch_size以适应更多样本
            learning_rate = 5e-4  # 降低学习率以获得更稳定的训练
            
            # 开始训练
            detector.train_lora(
                train_dataset=dataset,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_split=0.2  # 20%用于验证
            )
            
            print("\n训练完成！最佳模型权重已保存到 best_lora_weights.pth")
            
        else:
            # 视频分析模式
            print("\n扫描视频文件夹...")
            video_files = [f for f in os.listdir(detector.video_folder) if f.endswith('.mp4')]
            print(f"找到 {len(video_files)} 个视频文件")
            
            if not video_files:
                print("错误：未找到视频文件")
                return
                
            # 显示可用的视频文件
            print("\n=== 可用的视频文件 ===")
            for i, video in enumerate(video_files, 1):
                print(f"{i}. {video}")
                
            # 选择视频文件
            while True:
                try:
                    choice = int(input("\n请选择要分析的视频文件编号："))
                    if 1 <= choice <= len(video_files):
                        break
                    print("无效的选择，请重试")
                except ValueError:
                    print("请输入有效的数字")
                    
            video_name = video_files[choice - 1]
            
            # 询问是否使用训练好的LoRA权重
            use_lora = input("\n是否使用训练好的LoRA权重？(y/n): ").lower() == 'y'
            if use_lora:
                detector.load_lora_weights()
            
            # 分析视频
            print(f"\n开始分析视频：{video_name}")
            report = detector.analyze_video(video_name)
            
            if report:
                print("\n=== 视频信息 ===")
                print(f"分辨率: {report['video_info']['resolution']}")
                print(f"帧率: {report['video_info']['fps']} fps")
                print(f"总帧数: {report['video_info']['frame_count']}")
                
                print("\n=== 分析结果 ===")
                summary = detector.get_summary(report)
                print(summary)
                
                print("\n分析报告已保存到results文件夹")
            else:
                print("\n视频分析失败")
                
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        print("错误详情:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 