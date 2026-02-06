import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import transforms
from PIL import Image
from models.semi_supervised_model import SemiSupervisedPlantModel

def load_model(model_path, num_classes):
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SemiSupervisedPlantModel(num_classes=num_classes, pretrained=False)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    class_to_idx = checkpoint['class_to_idx']
    
    # 创建索引到类名的映射
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    model.to(device)
    model.eval()
    
    return model, idx_to_class, device

def predict_image(image_path, model, idx_to_class, device):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        logits, _ = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
    
    # 获取预测结果
    predicted_class = idx_to_class[predicted_idx.item()]
    confidence = confidence.item() * 100
    
    return predicted_class, confidence

def batch_predict(image_dir, model, idx_to_class, device):
    results = []
    
    # 遍历目录中的所有图片
    for img_name in os.listdir(image_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_dir, img_name)
            predicted_class, confidence = predict_image(img_path, model, idx_to_class, device)
            results.append((img_name, predicted_class, confidence))
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plant Identification Predictor')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='Path to trained model')
    parser.add_argument('--image_path', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of plant classes')
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    model, idx_to_class, device = load_model(args.model_path, args.num_classes)
    print(f"Model loaded successfully. Using device: {device}")
    print(f"Classes: {list(idx_to_class.values())}")
    
    if args.image_path:
        # 预测单张图片
        print(f"\nPredicting image: {args.image_path}")
        predicted_class, confidence = predict_image(args.image_path, model, idx_to_class, device)
        print(f"Predicted: {predicted_class} with confidence: {confidence:.2f}%")
    
    elif args.image_dir:
        # 批量预测
        print(f"\nPredicting images in directory: {args.image_dir}")
        results = batch_predict(args.image_dir, model, idx_to_class, device)
        
        print("\nPrediction results:")
        print("-" * 60)
        print(f"{'Image':<20} {'Predicted Class':<25} {'Confidence':<10}")
        print("-" * 60)
        
        for img_name, predicted_class, confidence in results:
            print(f"{img_name:<20} {predicted_class:<25} {confidence:.2f}%")
    
    else:
        print("Please provide either --image_path or --image_dir argument")
