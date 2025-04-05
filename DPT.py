import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DPTForSemanticSegmentation, DPTFeatureExtractor
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# Директории с изображениями и масками
images_dir = 'D:/pycharmprojects/Images/dataset/plots'
masks_dir_axes = 'D:/pycharmprojects/Images/dataset/masks/axes'
masks_dir_curves = 'D:/pycharmprojects/Images/dataset/masks/curves'
output_size = (128, 128)

# Классы
classes = [
    "curve", "curve2", "curve3", "curve4",
    "curve5", "curve6", "curve7", "curve8", "curve9", "curve10", "legend"
]
num_classes = len(classes)

# ======== Загрузка модели и feature extractor ========
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-hybrid-midas", num_labels=num_classes,  ignore_mismatched_sizes=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ======== Преобразование данных в формат PyTorch ========
class SegmentationDataset(Dataset):
    def __init__(self, images, masks, feature_extractor):
        self.images = images
        self.masks = masks
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Обрабатываем изображение через feature_extractor
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (1, C, H, W) -> (C, H, W)

        mask = torch.tensor(np.argmax(mask, axis=-1), dtype=torch.long)  # Индексная маска (H, W)

        return pixel_values, mask


# ======== Подготовка данных ========
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


# ======== Обучение модели ========
for i in range(1, 11):
    print(f"Training on dataset part {i}")

    images, masks = prepare_data(i)
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2)

    train_dataset = SegmentationDataset(X_train, y_train, feature_extractor)
    val_dataset = SegmentationDataset(X_val, y_val, feature_extractor)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    if i > 1:
        model.load_state_dict(torch.load(f'DPT/dpt_model_part{i - 1}.pth'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    best_loss = float('inf')

    for epoch in range(10):
        model.train()
        epoch_loss = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            outputs_resized = F.interpolate(outputs, size=(128, 128), mode="bilinear", align_corners=False)
            loss = criterion(outputs_resized, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'DPT/dpt_model_part{i}.pth')

    # Оценка на валидации
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    print(f"Validation Loss after Epoch {epoch + 1}: {val_loss / len(val_loader)}")


# ======== Визуализация метрик ========
train_losses = []
val_losses = []

for i in range(1, 11):
    model_path = f'DPT/dpt_model_part{i}.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(pixel_values=images).logits
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))
        train_losses.append(best_loss)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Loss over training")

plt.subplot(1, 2, 2)
plt.plot([1 - x for x in val_losses], label="1 - Validation Loss")
plt.legend()
plt.title("Improvement in Validation Score")
plt.show()
