# 植物识别AI系统

这是一个基于半监督学习的植物识别AI系统，支持识别几十种植物。系统使用PyTorch和ResNet18预训练模型，通过伪标签技术利用未标记数据提高模型性能。

## 项目结构

```
python_project/
├── data/
│   ├── labeled/       # 标记数据（按类别文件夹组织）
│   ├── unlabeled/     # 未标记数据
│   └── test/          # 测试数据
├── models/            # 模型保存目录
├── scripts/
│   ├── train.py       # 训练脚本
│   └── predict.py     # 预测脚本
├── utils/
│   └── data_loader.py # 数据加载模块
└── README.md          # 项目说明文档
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- torchvision
- Pillow
- tqdm

## 依赖安装

```bash
# 安装PyTorch和torchvision
pip install torch torchvision

# 安装其他依赖
pip install pillow tqdm
```

## 数据准备

1. **标记数据**：将带有标签的植物图片按类别组织到 `data/labeled/` 目录下，每个类别一个文件夹。
   ```
data/labeled/
├── 玫瑰/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── 向日葵/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── ...
   ```

2. **未标记数据**：将未标记的植物图片直接放入 `data/unlabeled/` 目录。
   ```
data/unlabeled/
├── 1.jpg
├── 2.jpg
└── ...
   ```

3. **测试数据**：将测试图片按类别组织到 `data/test/` 目录，结构与标记数据相同。

## 模型训练

### 基本用法

```bash
python scripts/train.py
```

### 训练参数说明

- `labeled_dir`：标记数据目录，默认 `data/labeled`
- `unlabeled_dir`：未标记数据目录，默认 `data/unlabeled`
- `test_dir`：测试数据目录，默认 `data/test`
- `num_classes`：植物类别数，默认 `10`（根据实际情况调整）
- `epochs`：训练轮数，默认 `50`
- `batch_size`：批次大小，默认 `32`
- `lr`：学习率，默认 `0.001`
- `threshold`：伪标签置信度阈值，默认 `0.9`
- `save_path`：模型保存路径，默认 `models/best_model.pth`

### 示例

```bash
python scripts/train.py --num_classes 20 --epochs 100 --batch_size 16
```

## 模型预测

### 预测单张图片

```bash
python scripts/predict.py --image_path path/to/image.jpg --num_classes 20
```

### 批量预测

```bash
python scripts/predict.py --image_dir path/to/images/ --num_classes 20
```

### 预测参数说明

- `model_path`：训练好的模型路径，默认 `models/best_model.pth`
- `image_path`：单张图片路径
- `image_dir`：图片目录路径
- `num_classes`：植物类别数，默认 `10`（与训练时保持一致）

## 模型说明

- **基础模型**：使用ResNet18预训练模型作为特征提取器
- **半监督方法**：采用伪标签技术，对未标记数据中置信度高的样本生成伪标签并参与训练
- **数据增强**：包括随机裁剪、翻转、旋转等操作，提高模型泛化能力
- **模型保存**：训练过程中会自动保存验证集上表现最好的模型

## 性能评估

训练过程中会实时输出以下指标：
- 标记数据的损失和准确率
- 伪标记数据的损失和准确率
- 测试集的准确率

最佳模型会保存在 `models/best_model.pth` 文件中，包含模型权重和类别映射信息。

## 注意事项

1. **数据质量**：标记数据的质量对模型性能影响很大，请确保标签的准确性
2. **数据量**：建议每个类别的标记数据至少有20-30张图片
3. **计算资源**：训练过程会使用GPU（如果可用），否则使用CPU
4. **模型调优**：根据实际数据情况，可以调整学习率、批次大小、置信度阈值等参数

## 扩展建议

1. **增加类别**：只需在 `data/labeled/` 目录下添加新的类别文件夹即可
2. **模型升级**：可以尝试使用ResNet34、ResNet50等更深的模型提高性能
3. **数据增强**：可以添加更多的数据增强方法，如颜色抖动、高斯模糊等
4. **迁移学习**：如果有领域特定的预训练模型，可以替换默认的ResNet18

## 常见问题

### Q: 训练时提示找不到数据？
A: 请检查数据目录结构是否正确，确保 `data/labeled/`、`data/unlabeled/` 和 `data/test/` 目录存在且包含正确的图片文件。

### Q: 预测时类别映射错误？
A: 请确保 `--num_classes` 参数与训练时使用的类别数一致，并且模型文件包含正确的类别映射信息。

### Q: 模型性能不佳？
A: 可以尝试以下方法：
- 增加标记数据量
- 调整训练参数（学习率、批次大小等）
- 使用更深的模型
- 增加数据增强强度
- 调整伪标签置信度阈值

## 许可证

本项目采用MIT许可证。
