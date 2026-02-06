import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SemiSupervisedPlantModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(SemiSupervisedPlantModel, self).__init__()
        
        # 使用ResNet18作为特征提取器
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.backbone = resnet18(weights=weights)
        else:
            self.backbone = resnet18()
        
        # 冻结预训练权重
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 替换最后的全连接层
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 移除原有的全连接层
        
        # 添加新的分类器
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 特征提取器的输出维度
        self.feature_dim = num_ftrs
    
    def forward(self, x):
        # 提取特征
        features = self.backbone(x)
        # 分类
        logits = self.classifier(features)
        return logits, features
    
    def get_features(self, x):
        # 仅提取特征
        features = self.backbone(x)
        return features
    
    def enable_backbone_grad(self):
        # 启用骨干网络的梯度计算（微调时使用）
        for param in self.backbone.parameters():
            param.requires_grad = True

class PseudoLabelGenerator:
    def __init__(self, threshold=0.9):
        self.threshold = threshold
    
    def generate_pseudo_labels(self, model, unlabeled_loader, device):
        model.eval()
        pseudo_labels = []
        confident_samples = []
        
        with torch.no_grad():
            for batch in unlabeled_loader:
                batch = batch.to(device)
                logits, _ = model(batch)
                probabilities = torch.softmax(logits, dim=1)
                max_probs, preds = torch.max(probabilities, dim=1)
                
                # 筛选置信度高于阈值的样本
                confident_mask = max_probs > self.threshold
                if confident_mask.sum() > 0:
                    confident_samples.append(batch[confident_mask])
                    pseudo_labels.append(preds[confident_mask])
        
        if confident_samples:
            confident_samples = torch.cat(confident_samples)
            pseudo_labels = torch.cat(pseudo_labels)
            return confident_samples, pseudo_labels
        else:
            return None, None
