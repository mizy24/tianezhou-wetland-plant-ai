import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class PlantDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_labeled=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_labeled = is_labeled
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        if is_labeled:
            # 标记数据：按类别文件夹组织
            classes = sorted(os.listdir(root_dir))
            for idx, class_name in enumerate(classes):
                self.class_to_idx[class_name] = idx
                class_dir = os.path.join(root_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith(('.jpg', '.jpeg', '.png')):
                            self.image_paths.append(os.path.join(class_dir, img_name))
                            self.labels.append(idx)
        else:
            # 未标记数据：直接加载所有图片
            for img_name in os.listdir(root_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root_dir, img_name))
                    self.labels.append(-1)  # 未标记数据标签为-1
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_labeled:
            label = self.labels[idx]
            return image, label
        else:
            return image
    
    def get_class_to_idx(self):
        return self.class_to_idx

def get_data_loaders(labeled_dir, unlabeled_dir, test_dir, batch_size=32, img_size=224):
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    labeled_dataset = PlantDataset(labeled_dir, transform=train_transform, is_labeled=True)
    unlabeled_dataset = PlantDataset(unlabeled_dir, transform=train_transform, is_labeled=False)
    test_dataset = PlantDataset(test_dir, transform=test_transform, is_labeled=True)
    
    # 创建数据加载器
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return labeled_loader, unlabeled_loader, test_loader, labeled_dataset.get_class_to_idx()
