import json
import os
import shutil
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import glob

def read_annotations(with_hat_dir, without_hat_dir):
    """Read annotations from both directories and combine them"""
    annotations = []
    
    # Process with_hat annotations
    for json_file in glob.glob(os.path.join(with_hat_dir, "*.json")):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Add has_hat=1 for with_hat images
            data['has_hat'] = 1
            data['image'] = os.path.join('有帽', os.path.basename(data['image']))
            annotations.append(data)
    
    # Process without_hat annotations
    for json_file in glob.glob(os.path.join(without_hat_dir, "*.json")):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Add has_hat=0 for without_hat images
            data['has_hat'] = 0
            data['image'] = os.path.join('无帽', os.path.basename(data['image']))
            annotations.append(data)
    
    return annotations

def convert_annotations(with_hat_dir, without_hat_dir, output_dir):
    """Convert annotations to YOLO format"""
    # Create directories
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    
    # Read all annotations
    annotations = read_annotations(with_hat_dir, without_hat_dir)
    
    # Split data into train and validation sets
    train_data, val_data = train_test_split(annotations, test_size=0.2, random_state=42)
    
    def process_dataset(dataset, subset):
        processed_count = 0
        for item in dataset:
            try:
                # Get image path and normalize it
                img_path = os.path.join(os.path.dirname(with_hat_dir), item['image'])
                img_path = os.path.abspath(img_path)
                
                if not os.path.exists(img_path):
                    print(f"Warning: Image not found: {img_path}")
                    continue
                
                # Read image to get dimensions
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Warning: Could not read image: {img_path}")
                    continue
                
                h, w = img.shape[:2]
                
                # Generate a new filename using a counter to avoid Chinese characters
                new_img_name = f"{subset}_{processed_count:06d}.jpg"
                new_img_path = os.path.join(output_dir, 'images', subset, new_img_name)
                
                # Save image using cv2.imwrite
                cv2.imwrite(new_img_path, img)
                
                # Create YOLO format label
                label_name = f"{subset}_{processed_count:06d}.txt"
                label_path = os.path.join(output_dir, 'labels', subset, label_name)
                
                # For binary classification, we'll use class 0 for no_hat and class 1 for has_hat
                with open(label_path, 'w') as f:
                    if item['has_hat'] == 1:
                        f.write('1 0.5 0.5 1.0 1.0\n')  # has_hat
                    else:
                        f.write('0 0.5 0.5 1.0 1.0\n')  # no_hat
                
                processed_count += 1
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        
        return processed_count
    
    # Process train and validation sets
    train_processed = process_dataset(train_data, 'train')
    val_processed = process_dataset(val_data, 'val')
    
    # Create dataset.yaml with absolute path
    yaml_content = f"""path: {os.path.abspath(output_dir)}  # dataset root dir
train: images/train  # train images
val: images/val  # val images

# Classes
names:
  0: no_hat
  1: has_hat"""
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset prepared in {output_dir}")
    print(f"Training samples processed: {train_processed}")
    print(f"Validation samples processed: {val_processed}")

def prepare_dataset():
    # Create YOLO dataset structure
    os.makedirs('dataset/yolo_dataset/images/train', exist_ok=True)
    os.makedirs('dataset/yolo_dataset/images/val', exist_ok=True)
    os.makedirs('dataset/yolo_dataset/labels/train', exist_ok=True)
    os.makedirs('dataset/yolo_dataset/labels/val', exist_ok=True)
    
    # Get all annotation files from the new directory
    annotations_dir = r"D:\labels_my-project-name_2025-02-10-05-28-35"
    images_dir = r"C:\Users\dorot\Desktop\crawl爬虫\downloaded_images"
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
    
    # Split into train and validation sets
    train_files, val_files = train_test_split(annotation_files, test_size=0.2, random_state=42)
    
    def process_files(files, split='train'):
        processed = 0
        for ann_file in files:
            try:
                # Get corresponding image name
                img_name = ann_file.replace('.txt', '.jpg')  # or other image extension if different
                img_path = os.path.join(images_dir, img_name)
                
                # Read image using cv2.imdecode to handle potential special characters
                try:
                    img_path_bytes = img_path.encode('utf-8')
                    img = cv2.imdecode(np.fromfile(img_path_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        print(f"Warning: can't open/read file: {img_path}")
                        continue
                    
                    # Generate new filename
                    new_filename = f"image_{processed:06d}.jpg"
                    
                    # Save image
                    cv2.imwrite(f'dataset/yolo_dataset/images/{split}/{new_filename}', img)
                    
                    # Copy and rename annotation file
                    src_ann = os.path.join(annotations_dir, ann_file)
                    dst_ann = f'dataset/yolo_dataset/labels/{split}/{new_filename.replace(".jpg", ".txt")}'
                    shutil.copy2(src_ann, dst_ann)
                    
                    processed += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
                    
            except Exception as e:
                print(f"Error processing annotation {ann_file}: {str(e)}")
                continue
        
        return processed
    
    # Process training and validation files
    train_processed = process_files(train_files, 'train')
    val_processed = process_files(val_files, 'val')
    
    # Create dataset.yaml
    yaml_content = f"""path: {os.path.abspath('dataset/yolo_dataset')}
train: images/train
val: images/val

nc: 2
names: ['no_hat', 'has_hat']"""

    with open('dataset/yolo_dataset/dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset prepared in dataset/yolo_dataset")
    print(f"Training samples: {train_processed}")
    print(f"Validation samples: {val_processed}")

if __name__ == '__main__':
    prepare_dataset() 