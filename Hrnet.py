import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import cv2
import gc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import onnx2tf
import torch
from mmseg.models import build_segmentor
from mmengine.config import Config
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D

# Директории
images_dir = 'new_dataset/plots'
masks_dir_axes = 'new_dataset/masks/axes'
masks_dir_curves = 'new_dataset/masks/curves'
output_size = (128, 256)

# Классы
classes = [
    "curve", "curve2", "curve3", "curve4",
    "curve5", "curve6", "curve7", "curve8", "curve9", "curve10", "legend"
]
num_classes = len(classes)


# ======== HRNet Model ========
def convert_hrnet_to_tf():
    cfg = Config.fromfile('configs/hrnet/fcn_hr18_512x1024_80k_cityscapes.py')
    model = build_segmentor(cfg.model)
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 128)  # Входное изображение
    torch.onnx.export(model, dummy_input, "hrnet.onnx", opset_version=11)

    onnx2tf.convert(
        input_onnx_file_path="hrnet.onnx",
        output_folder_path="hrnet_tf"
    )


def load_hrnet_model():
    return tf.saved_model.load("hrnet_tf")


# ======== Metrics ========
def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((2. * intersection + smooth) / (union + smooth))


def iou_coefficient(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth))


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

        # Делаем количество каналов равным количеству классов
        while len(mask_channels) < num_classes:
            mask_channels.append(np.zeros(output_size, dtype=np.float32))

        combined_mask = np.stack(mask_channels, axis=-1)
        masks.append(combined_mask)

    return np.array(images), np.array(masks)


# ======== Training Loop ========
input_shape = (output_size[0], output_size[1], 3)

# Если HRNet ещё не конвертирован, конвертируем
convert_hrnet_to_tf()

# Загружаем конвертированную модель HRNet
hrnet = load_hrnet_model()

for i in range(1, 11):
    images, masks = prepare_data(i)
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    tf.keras.backend.clear_session()
    gc.collect()

    if i > 1:
        model = tf.keras.models.load_model(f'Hr/model_part{i - 1}.h5', custom_objects={
            "dice_coefficient": dice_coefficient, "iou_coefficient": iou_coefficient
        })
    else:
        inputs = tf.keras.Input(shape=input_shape)
        hrnet_output = hrnet(inputs)
        x = Conv2D(num_classes, (1, 1), activation="softmax")(hrnet_output)
        model = Model(inputs, x)

    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy',
                  metrics=[dice_coefficient, iou_coefficient])

    checkpoint = ModelCheckpoint(f'Hr/model_part{i}.h5', monitor='val_loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-7)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=4, epochs=100,
                        callbacks=[checkpoint, reduce_lr, early_stopping])

acc = history.history['accuracy']  # Точность на обучающей выборке
val_acc = history.history['val_accuracy']  # Точность на валидационной выборке

loss = history.history['loss']  # Потери на обучающей выборке
val_loss = history.history['val_loss']  # Потери на валидационной выборке

epochs_range = range(len(history.history['accuracy']))  # Диапазон эпох

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Точность обучения')  # График точности на обучающей выборке
plt.plot(epochs_range, val_acc, label='Точность валидации')  # График точности на валидационной выборке
plt.legend(loc='lower right')
plt.title('Точность обучения и валидации')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Потери обучения')  # График потерь на обучающей выборке
plt.plot(epochs_range, val_loss, label='Потери валидации')  # График потерь на валидационной выборке
plt.legend(loc='upper right')
plt.title('Потери обучения и валидации')
plt.savefig('Hr/training_results.png', dpi=300, bbox_inches='tight')
