import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from mmcls.models import build_backbone
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# Директории с изображениями и масками
images_dir = 'D:/pycharmprojects/Images/dataset/plots'
masks_dir_axes = 'D:/pycharmprojects/Images/dataset/masks/axes'
masks_dir_curves = 'D:/pycharmprojects/Images/dataset/masks/curves'
output_size = (128, 128)
classes = [
    "curve", "curve2", "curve3", "curve4",
    "curve5", "curve6", "curve7", "curve8", "curve9", "curve10", "legend"
]

# Конфиг для HRNet
cfg = dict(
    type='HRNet',
    extra=dict(
        stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', num_blocks=(4,), num_channels=(64,)),
        stage2=dict(num_modules=1, num_branches=2, block='BASIC', num_blocks=(4, 4), num_channels=(32, 64)),
        stage3=dict(num_modules=4, num_branches=3, block='BASIC', num_blocks=(4, 4, 4), num_channels=(32, 64, 128)),
        stage4=dict(num_modules=3, num_branches=4, block='BASIC', num_blocks=(4, 4, 4, 4),
                    num_channels=(32, 64, 128, 256))
    )
)

# Создаём HRNet backbone
model = build_backbone(cfg)
model.init_weights()


# Преобразуем данные в формат PyTorch
class SegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Преобразуем в тензор
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # (H, W, C) -> (C, H, W)

        # Для CrossEntropyLoss маска должна быть индексовой (для каждого пикселя класс)
        mask = torch.tensor(np.argmax(mask, axis=-1), dtype=torch.long)  # Преобразуем в индексный массив (H, W)

        return image, mask



# ======== Data Preparation ========
def prepare_data(total):
    task_files = {}
    images, masks = [], []
    for index in range(1, 1001):
        task_files[(total, index)] = {"images": [], "masks": []}
        task_files[(total, index)]["images"].append(f'{images_dir}/index_{index}_with_{total}_curves.png')
        for curve in range(1, total + 1):
            task_files[(total, index)]["masks"].append(
                f'{masks_dir_curves}/sample_{index}_curve_{curve}_of_{total}.png')
        task_files[(total, index)]["masks"].append(f'{masks_dir_axes}/axes_sample_{index}_with_{total}_curves.png')

    for key, files in tqdm(task_files.items(), desc="Loading data"):
        for image_file in files["images"]:
            img = cv2.imread(image_file)
            img = cv2.resize(img, output_size)
            img = img / 255.0
            images.append(img)
        mask_channels = []
        for mask_file in files["masks"]:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, output_size)
            mask = (mask > 0).astype(np.float32)
            mask_channels.append(mask)
        for i in range(len(files['masks']) - 1, 10):
            mask = np.zeros(output_size, dtype=np.float32)
            mask_channels.append(mask)
        mask_file = files["masks"][-1]
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, output_size)
        mask = (mask > 0).astype(np.float32)
        combined_mask = np.stack(mask_channels, axis=-1)
        masks.append(combined_mask)
    return np.array(images), np.array(masks)



# ======== Define HRNet-based model for segmentation ========
class HRNetSegmentation(nn.Module):
    def __init__(self, hrnet_model, num_classes):
        super(HRNetSegmentation, self).__init__()
        self.hrnet = hrnet_model
        self.conv1x1 = nn.Conv2d(256, num_classes, kernel_size=1)  # Для вывода классов

    def forward(self, x):
        # Проходим через HRNet
        features = self.hrnet(x)
        output = self.conv1x1(features[-1])  # Используем последний уровень признаков

        # Применяем интерполяцию для масштабирования результата
        output = F.interpolate(output, size=(128, 128), mode='bilinear', align_corners=False)

        return output

# ======== Dice Coefficient Function ========
def dice_coefficient(preds, targets, smooth=1e-6):
    """Вычисление коэффициента Dice для бинарной сегментации."""
    intersection = (preds & targets).float().sum((1, 2))  # Суммируем по каждой паре изображений
    union = preds.float().sum((1, 2)) + targets.float().sum((1, 2))  # Объединение
    dice = (2. * intersection + smooth) / (union + smooth)  # Формула для вычисления Dice
    return dice.mean()  # Возвращаем среднее значение по батчу


# ======== Training Loop ========

input_shape = (3, 128, 128)
num_classes = len(classes)

# История для визуализации метрик
train_losses = []
val_losses = []
train_dice_coeffs = []
val_dice_coeffs = []

# Цикл обучения по 10 частям данных
for i in range(1, 11):
    # Загружаем и готовим данные
    images, masks = prepare_data(i)
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2)

    # Данные для PyTorch
    train_dataset = SegmentationDataset(X_train, y_train)
    val_dataset = SegmentationDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Модель HRNet
    if i == 1:
        model = HRNetSegmentation(model, num_classes)
    else:
        model.load_state_dict(torch.load(f'Hrnet/hrnet_model_part{i - 1}.pth'))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    best_loss = float('inf')

    # Обучение модели
    for epoch in range(10):  # Период тренировки
        model.train()
        epoch_loss = 0
        epoch_dice_coeff = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, masks = images.cuda(), masks.cuda()

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Вычисление Dice коэффициента для текущей партии
            outputs = torch.argmax(outputs, dim=1)  # Получаем индексы предсказанных классов
            dice = dice_coefficient(outputs, masks)  # Функция для вычисления Dice
            epoch_dice_coeff += dice.item()

        # Сохраняем метрики для графика
        train_losses.append(epoch_loss / len(train_loader))
        train_dice_coeffs.append(epoch_dice_coeff / len(train_loader))

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}, Dice: {epoch_dice_coeff / len(train_loader)}')

        # Сохранение модели
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'Hrnet/hrnet_model_part{i}.pth')

    # Оценка на валидации
    model.eval()
    val_loss = 0
    val_dice_coeff = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            # Вычисление Dice коэффициента для валидации
            outputs = torch.argmax(outputs, dim=1)  # Получаем индексы предсказанных классов
            dice = dice_coefficient(outputs, masks)  # Функция для вычисления Dice
            val_dice_coeff += dice.item()

    val_losses.append(val_loss / len(val_loader))
    val_dice_coeffs.append(val_dice_coeff / len(val_loader))

    print(f"Validation Loss after Epoch {epoch + 1}: {val_loss / len(val_loader)}, Dice: {val_dice_coeff / len(val_loader)}")

# ======== Plot Metrics ========
# Визуализация графиков метрик

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Loss")
plt.subplot(1, 2, 2)
plt.plot(train_dice_coeffs, label="Train Dice Coefficient")
plt.plot(val_dice_coeffs, label="Validation Dice Coefficient")
plt.legend()
plt.title("Dice Coefficient")
plt.show()


