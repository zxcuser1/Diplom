import logging
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential

# Отключение предупреждений TensorFlow
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Путь к данным
gr = Path('processed_dataset')

# Параметры данных
batch_size = 32
img_width, img_height = 256, 256

# Создание обучающего набора данных с разделением на обучающую и валидационную выборки
train_ds = tf.keras.utils.image_dataset_from_directory(
    gr,
    validation_split=0.2,  # 20% данных будут использованы для валидации
    subset="training",
    seed=123,  # Для воспроизводимости результатов
    image_size=(img_height, img_width),  # Размер изображений
    batch_size=batch_size)

# Создание валидационного набора данных
val_ds = tf.keras.utils.image_dataset_from_directory(
    gr,
    validation_split=0.2,  # 20% данных будут использованы для валидации
    subset="validation",
    seed=123,  # Для воспроизводимости результатов
    image_size=(img_height, img_width),  # Размер изображений
    batch_size=batch_size
)

# Получение списка классов (меток)
class_names = train_ds.class_names
print(f"Class names: {class_names}")

# Оптимизация производительности при работе с данными
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Определение формы входных данных для модели
IMG_SHAPE = (img_height, img_width, 3)

# Используем функцию предварительной обработки для VGG16
preprocess_input = tf.keras.applications.vgg16.preprocess_input

# Загрузка предобученной модели MobileNetV2 без верхних слоев и с весами из ImageNet
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,  # Не включаем полносвязные слои на конце
                                               weights='imagenet')

# Замораживаем веса предобученной модели, чтобы они не изменялись во время обучения
base_model.trainable = False

# Определение слоев для аугментации данных
data_augmentation = Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    # Случайное горизонтальное отражение
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1)  # Случайный поворот изображений
])

# Создание модели
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)  # Применение аугментации данных
x = preprocess_input(x)  # Предварительная обработка входных данных
x = base_model(x, training=False)  # Пропуск данных через базовую модель
x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Глобальный усредненный пуллинг
x = tf.keras.layers.Dropout(0.1)(x)  # Dropout для регуляризации
outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(
    x)  # Выходной слой с softmax для многоклассовой классификации
model = tf.keras.Model(inputs, outputs)

# Компиляция модели
model.compile(
    optimizer='adam',  # Оптимизатор Adam
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Функция потерь для многоклассовой классификации
    metrics=['accuracy']  # Метрика точности
)

# Обучение модели
epochs = 10
history = model.fit(
    train_ds,  # Обучающая выборка
    validation_data=val_ds,  # Валидационная выборка
    batch_size=batch_size,  # Размер пакета
    epochs=epochs  # Количество эпох
)

# Визуализация результатов обучения
acc = history.history['accuracy']  # Точность на обучающей выборке
val_acc = history.history['val_accuracy']  # Точность на валидационной выборке

loss = history.history['loss']  # Потери на обучающей выборке
val_loss = history.history['val_loss']  # Потери на валидационной выборке

epochs_range = range(epochs)  # Диапазон эпох

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
plt.show()

# Сохранение модели
model.save('model/VIT/VIT_model_graphics.h5')
print('Model saved')
