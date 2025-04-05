import gc

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.saving.save import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tqdm import tqdm

images_dir = 'new_dataset/plots'
masks_dir_axes = 'new_dataset/masks/axes'
masks_dir_curves = 'new_dataset/masks/curves'
output_size = (256, 128)
classes = [
    "curve", "curve2", "curve3", "curve4",
    "curve5", "curve6", "curve7", "curve8", "curve9", "curve10", "legend"
]


# ======== DeepLabV3+ Model ========
def build_deeplabv3(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = inputs

    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)

    x = base_model.output
    x = layers.Conv2D(256, (1, 1), activation="relu")(x)

    # Upsampling до исходного размера
    x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)
    x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

    x = Conv2D(256, (1, 1), activation="relu", kernel_initializer=HeNormal())(x)

    model = Model(inputs=inputs, outputs=x)
    return model


# ======== Data Preparation ========
def prepare_data1(total):
    task_files = {}
    images, masks = [], []

    for index in range(1, 1001):
        task_files[(total, index)] = {"images": [], "masks": []}
        task_files[(total, index)]["images"].append(f'{images_dir}/index_{index}_with_{total}_curves.png')
        for curve in range(1, total + 1):
            task_files[(total, index)]["masks"].append(
                f'{masks_dir_curves}/sample_{index}_curve_{curve}_of_{total}.png')
        task_files[(total, index)]["masks"].append(f'{masks_dir_axes}/legend_sample_{index}_with_{total}_curves.png')

    for key, files in tqdm(task_files.items(), desc="Loading data"):
        for image_file in files["images"]:
            img = cv2.imread(image_file)
            img = cv2.resize(img, (output_size[1], output_size[0]))
            img = img / 255.0
            images.append(img)

        mask_channels = []
        for mask_file in files["masks"]:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (output_size[1], output_size[0]))
            mask = (mask > 0).astype(np.float32)
            mask_channels.append(mask)

        # Проверка размера масок
        # print(f"Размеры маски: {[mask.shape for mask in mask_channels]}")

        # Делаем количество каналов равным количеству классов
        while len(mask_channels) < num_classes:
            mask_channels.append(np.zeros((output_size[0], output_size[1]), dtype=np.float32))

        combined_mask = np.stack(mask_channels, axis=-1)
        masks.append(combined_mask)

    # print(f"Размер итогового массива масок: {np.array(masks).shape}")
    return np.array(images), np.array(masks)


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
            img = cv2.resize(img, (output_size[1], output_size[0]))
            img = img / 255.0
            images.append(img)

        mask_channels = []
        for mask_file in files["masks"]:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (output_size[1], output_size[0]))
            mask = (mask > 0).astype(np.float32)
            mask_channels.append(mask)

        # Проверка размера масок
        #print(f"Размеры маски: {[mask.shape for mask in mask_channels]}")

        # Делаем количество каналов равным количеству классов
        while len(mask_channels) < num_classes:
            mask_channels.append(np.zeros((output_size[0], output_size[1]), dtype=np.float32))

        combined_mask = np.stack(mask_channels, axis=-1)
        masks.append(combined_mask)

    #print(f"Размер итогового массива масок: {np.array(masks).shape}")
    return np.array(images), np.array(masks)


# ======== Training Loop ========
input_shape = (output_size[0], output_size[1], 3)
num_classes = len(classes)

for i in range(1, 11):
    images, masks = prepare_data(i)
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2)

    y_train = np.argmax(y_train, axis=-1)
    y_val = np.argmax(y_val, axis=-1)

    tf.keras.backend.clear_session()
    gc.collect()

    if i == 1:
        model = build_deeplabv3(input_shape, num_classes)
    else:
        model = load_model(f'deepv3/model_part{i - 1}.h5')

    model.compile(
        optimizer='adam',  # Оптимизатор Adam
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])  # Метрика точности

    checkpoint = ModelCheckpoint(f'deepv3/model_part{i}.h5', monitor='val_loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=1e-7)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=4, epochs=100,
                        callbacks=[checkpoint, reduce_lr, early_stopping])

# Визуализация результатов обучения

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
plt.savefig('deepv3/training_results.png', dpi=300, bbox_inches='tight')

images_dir = 'new_dataset/dataset_curves/plots'
masks_dir_axes = 'new_dataset/dataset_curves/masks/legend'
masks_dir_curves = 'new_dataset/dataset_curves/masks/curves'

images, masks = prepare_data1(1)
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2)

y_train = np.argmax(y_train, axis=-1)
y_val = np.argmax(y_val, axis=-1)

tf.keras.backend.clear_session()
gc.collect()

model = load_model(f'deepv3/model_part{10}.h5')

model.compile(
    optimizer='adam',  # Оптимизатор Adam
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])  # Метрика точности

checkpoint = ModelCheckpoint(f'deepv3/trainded_model.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=1e-7)

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
plt.savefig('deepv3/training_results1.png', dpi=300, bbox_inches='tight')
