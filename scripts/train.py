import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.data_loader import get_data_loaders
from models.semi_supervised_model import SemiSupervisedPlantModel, PseudoLabelGenerator

def load_checkpoint_info(checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        class_to_idx = checkpoint.get('class_to_idx', None)
        if class_to_idx:
            num_classes = len(class_to_idx)
            return num_classes, best_accuracy, class_to_idx
    return None, None, None

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        class_to_idx = checkpoint.get('class_to_idx', None)
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Best accuracy from checkpoint: {best_accuracy:.2f}%")
        return best_accuracy, class_to_idx
    return None, None

def train_semi_supervised(labeled_dir, unlabeled_dir, test_dir, num_classes, epochs=50, batch_size=32, lr=0.001, threshold=0.9, save_path='models/best_model.pth', resume_from=None):
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 检查是否要从已有模型加载
    checkpoint_num_classes = None
    checkpoint_best_accuracy = 0.0
    checkpoint_class_to_idx = None
    
    if resume_from and os.path.exists(resume_from):
        checkpoint_num_classes, checkpoint_best_accuracy, checkpoint_class_to_idx = load_checkpoint_info(resume_from)
        if checkpoint_num_classes:
            num_classes = checkpoint_num_classes
            print(f"Resuming from checkpoint with {num_classes} classes")
    elif os.path.exists(save_path):
        checkpoint_num_classes, checkpoint_best_accuracy, checkpoint_class_to_idx = load_checkpoint_info(save_path)
        if checkpoint_num_classes:
            num_classes = checkpoint_num_classes
            print(f"Found existing model with {num_classes} classes")
    
    # 加载数据
    labeled_loader, unlabeled_loader, test_loader, class_to_idx = get_data_loaders(
        labeled_dir, unlabeled_dir, test_dir, batch_size=batch_size
    )
    print(f"Class to index mapping: {class_to_idx}")
    
    # 初始化模型
    model = SemiSupervisedPlantModel(num_classes=num_classes, pretrained=True)
    model.to(device)
    
    # 初始化伪标签生成器
    pseudo_label_generator = PseudoLabelGenerator(threshold=threshold)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0.0
    
    # 尝试从已有模型加载
    if resume_from and os.path.exists(resume_from):
        loaded_accuracy, loaded_class_to_idx = load_checkpoint(model, optimizer, resume_from)
        if loaded_accuracy is not None:
            best_accuracy = loaded_accuracy
            print(f"Resuming training from {resume_from}")
            print(f"Starting with best accuracy: {best_accuracy:.2f}%")
    elif os.path.exists(save_path) and checkpoint_num_classes:
        loaded_accuracy, loaded_class_to_idx = load_checkpoint(model, optimizer, save_path)
        if loaded_accuracy is not None:
            best_accuracy = loaded_accuracy
            print(f"Found existing model at {save_path}")
            print(f"Starting with best accuracy: {best_accuracy:.2f}%")
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("=" * 50)
        
        # 1. 训练标记数据
        print("Training on labeled data...")
        for images, labels in tqdm(labeled_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        labeled_accuracy = 100 * correct / total
        labeled_loss = running_loss / len(labeled_loader)
        print(f"Labeled data - Loss: {labeled_loss:.4f}, Accuracy: {labeled_accuracy:.2f}%")
        
        # 2. 生成伪标签并训练未标记数据
        print("Generating pseudo labels...")
        pseudo_images, pseudo_labels = pseudo_label_generator.generate_pseudo_labels(
            model, unlabeled_loader, device
        )
        
        if pseudo_images is not None and pseudo_labels is not None:
            print(f"Found {pseudo_images.size(0)} confident unlabeled samples")
            print("Training on pseudo labeled data...")
            
            # 训练伪标记数据
            optimizer.zero_grad()
            logits, _ = model(pseudo_images)
            loss = criterion(logits, pseudo_labels)
            loss.backward()
            optimizer.step()
            
            pseudo_loss = loss.item()
            _, predicted = torch.max(logits.data, 1)
            pseudo_correct = (predicted == pseudo_labels).sum().item()
            pseudo_accuracy = 100 * pseudo_correct / pseudo_images.size(0)
            print(f"Pseudo labeled data - Loss: {pseudo_loss:.4f}, Accuracy: {pseudo_accuracy:.2f}%")
        else:
            print("No confident unlabeled samples found")
        
        # 3. 评估模型
        test_accuracy = evaluate_model(model, test_loader, device)
        print(f"Test accuracy: {test_accuracy:.2f}%")
        
        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': class_to_idx,
                'best_accuracy': best_accuracy
            }, save_path)
            print(f"Saved best model with accuracy: {best_accuracy:.2f}%")
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    return best_accuracy

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Semi-supervised Plant Classification Trainer')
    parser.add_argument('--labeled_dir', type=str, default='data/labeled', help='Labeled data directory')
    parser.add_argument('--unlabeled_dir', type=str, default='data/unlabeled', help='Unlabeled data directory')
    parser.add_argument('--test_dir', type=str, default='data/test', help='Test data directory')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of plant classes')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--threshold', type=float, default=0.9, help='Pseudo label confidence threshold')
    parser.add_argument('--save_path', type=str, default='models/best_model.pth', help='Path to save best model')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    train_semi_supervised(
        labeled_dir=args.labeled_dir,
        unlabeled_dir=args.unlabeled_dir,
        test_dir=args.test_dir,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        threshold=args.threshold,
        save_path=args.save_path,
        resume_from=args.resume_from
    )
