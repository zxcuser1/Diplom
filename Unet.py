import gc
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tqdm import tqdm

output_size = (256, 256)
classes = [
    "curve", "curve2", "curve3", "curve4",
    "curve5", "curve6", "curve7", "curve8", "curve9", "curve10", "legend"
]

physical_devices = tf.config.list_physical_devices('GPU')

if physical_devices:

    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
    )


# ======== U-Net Model ========
def build_unet(input_shape, num_classes):
    inputs = layers.Input(shape=(input_shape[0], input_shape[1], 1))

    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        return x

    def upsample_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
        x = layers.concatenate([x, skip])
        return conv_block(x, filters)

    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = conv_block(p4, 1024)

    u6 = upsample_block(c5, c4, 512)
    u7 = upsample_block(u6, c3, 256)
    u8 = upsample_block(u7, c2, 128)
    u9 = upsample_block(u8, c1, 64)

    outputs = layers.Conv2D(num_classes, (1, 1))(u9)
    return models.Model(inputs, outputs)


def prepare_data(total, first=1, last=1000, flag=False, augment=False):
    images, masks = [], []

    for index in tqdm(range(first, last + 1), desc="Loading images"):
        # Загружаем изображение графика
        image_path = f'{images_dir}/index_{index}_with_{total}_curves.png'
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (output_size[1], output_size[0]))
        img = img / 255.0  # Нормализация

        # Создаем многоканальную маску
        mask_channels = np.zeros((output_size[0], output_size[1]))

        # Загружаем маски кривых (уникальные классы)
        for curve in range(1, total + 1):
            mask_path = f'{masks_dir_curves}/sample_{index}_curve_{curve}_of_{total}.png'
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (output_size[1], output_size[0]))
            mask = mask / 255.0
            for i in range(0, 256):
                for j in range(0, 256):
                    mask[i][j] *= curve

            for i in range(0, 256):
                for j in range(256):
                    if mask_channels[i][j] == 0 and mask[i][j] != 0:
                        mask_channels[i][j] = mask[i][j]

        mask_axes = None
        if flag:
            mask_axes = cv2.imread(f'{masks_dir_axes}/legend_sample_{index}_with_{total}_curves.png',
                                   cv2.IMREAD_GRAYSCALE)
        else:
            mask_axes = cv2.imread(f'{masks_dir_axes}/axes_sample_{index}_with_{total}_curves.png',
                                   cv2.IMREAD_GRAYSCALE)
        mask_axes = cv2.resize(mask_axes, (output_size[1], output_size[0]))
        mask_axes = mask_axes / 255.0
        mask_axes = (mask_axes > 0).astype(np.uint8) * 11
        mask_channels = np.maximum(mask_channels, mask_axes)

        if augment:
            rotations = [90, 180, 270]
            for k in rotations:
                rotated_img = cv2.rotate(img, {0: cv2.ROTATE_90_CLOCKWISE,
                                               90: cv2.ROTATE_90_COUNTERCLOCKWISE,
                                               180: cv2.ROTATE_180,
                                               270: cv2.ROTATE_90_COUNTERCLOCKWISE}[k])

                rotated_mask = cv2.rotate(mask_channels, {0: cv2.ROTATE_90_CLOCKWISE,
                                                          90: cv2.ROTATE_90_COUNTERCLOCKWISE,
                                                          180: cv2.ROTATE_180,
                                                          270: cv2.ROTATE_90_COUNTERCLOCKWISE}[k])

                images.append(rotated_img)
                masks.append(rotated_mask)

        images.append(img)
        masks.append(mask_channels)

    return np.array(images), np.array(masks)


history = []
images_dir = 'new_dataset/dataset_curves/intersection/plots'
masks_dir_axes = 'new_dataset/dataset_curves/intersection/masks/legend'
masks_dir_curves = 'new_dataset/dataset_curves/intersection/masks/curves'
model = build_unet(input_shape=(256, 256), num_classes=12)

for i in range(2, 11):
    print(f'NOW {i}')
    for j in range(1, 1000, 1000):
        tf.keras.backend.clear_session()
        gc.collect()
        images, masks = prepare_data(total=i, flag=True, augment=False)
        X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2)
        optimizer = Adam(learning_rate=1e-5)  # Уменьшаем скорость обучения
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        history.append(model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=2, epochs=10))

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
plt.savefig('Unet256/training_results3 .png', dpi=300, bbox_inches='tight')
