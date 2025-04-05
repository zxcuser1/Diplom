import logging
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.regularizers import L2

# Отключение предупреждений TensorFlow
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Путь к данным
gr = Path('processed_dataset')

# Параметры данных
batch_size = 8
img_width, img_height = 128, 128

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

# Определение модели с помощью метода CNN
num_classes = len(class_names)  # Количество классов
model = Sequential(
    [
        # Аугментация данных
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        # Масштабирование значений пикселей
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),  # Случайные повороты изображений
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),  # Случайное увеличение/уменьшение изображений
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),  # Случайное изменение контраста изображений

        # Первый сверточный слой с регуляризацией L2
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=L2(1e-5)),
        tf.keras.layers.MaxPooling2D((3, 3)),  # Максимальное объединение
        tf.keras.layers.Dropout(0.2),  # Отсев

        # Второй сверточный слой с регуляризацией L2
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=L2(1e-5)),
        tf.keras.layers.MaxPooling2D(3, 3),  # Максимальное объединение
        tf.keras.layers.Dropout(0.2),  # Отсев

        # Третий сверточный слой с регуляризацией L2
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=L2(1e-5)),
        tf.keras.layers.MaxPooling2D(3, 3),  # Максимальное объединение
        tf.keras.layers.Dropout(0.2),  # Отсев

        # Полносвязный слой
        tf.keras.layers.Flatten(),  # Преобразование данных в одномерный массив
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=L2(1e-5)),
        # Полносвязный слой с регуляризацией L2
        tf.keras.layers.Dense(num_classes, activation='softmax')
        # Выходной слой с функцией активации softmax для многоклассовой классификации
    ]
)

# Компиляция модели
model.compile(
    optimizer='adam',  # Оптимизатор Adam
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # Функция потерь для многоклассовой классификации
    metrics=['accuracy'])  # Метрика точности

# Вывод структуры модели
model.summary()

# Обучение модели
epochs = 10
history = model.fit(
    train_ds,  # Обучающая выборка
    validation_data=val_ds,  # Валидационная выборка
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
model.save('CNN/CNN_model_graphics.h5')
print('Model saved')
