from pathlib import Path
import logging
import tensorflow as tf
from keras.regularizers import L2
from matplotlib import pyplot as plt
from keras.models import Sequential

# Отключение предупреждений TensorFlow
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Задаем путь к данным
gr = Path('validation_dataset')

# Задаем параметры
img_width, img_height = 256, 256  # Размеры изображений
batch_size = 32  # Размер батча
epochs = 15  # Количество эпох

# Загрузка тренировочного набора данных с разделением на тренировочную и валидационную выборки
train_ds = tf.keras.utils.image_dataset_from_directory(
    gr,
    validation_split=0.2,  # Процент данных для валидации
    subset="training",  # Указываем, что это тренировочный набор данных
    seed=123,  # Установка случайного семени для воспроизводимости
    image_size=(img_height, img_width),  # Изменение размера изображений
    batch_size=batch_size  # Размер батча
)

# Загрузка валидационного набора данных
val_ds = tf.keras.utils.image_dataset_from_directory(
    gr,
    validation_split=0.2,  # Процент данных для валидации
    subset="validation",  # Указываем, что это валидационный набор данных
    seed=123,  # Установка случайного семени для воспроизводимости
    image_size=(img_height, img_width),  # Изменение размера изображений
    batch_size=batch_size  # Размер батча
)

# Получение имен классов
class_names = train_ds.class_names
print(f"Class names: {class_names}")

# Кэширование и предзагрузка данных для ускорения обучения
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,  # Убираем стандартный классификатор
    weights="imagenet"
)

# Замораживаем предобученные веса
base_model.trainable = False

# Создаем модель
num_classes = len(class_names)
model = Sequential(
    [
        # Аугментация данных и нормализация изображений
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),

        # Предобученная модель
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),  # Теперь после base_model

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ]
)

# Компиляция модели
model.compile(
    optimizer='adam',  # Оптимизатор Adam
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Функция потерь
    metrics=['accuracy']  # Метрика для оценки качества модели
)

# Вывод структуры модели
model.summary()

# Обучение модели
history = model.fit(
    train_ds,  # Тренировочный набор данных
    validation_data=val_ds,  # Валидационный набор данных
    epochs=epochs  # Количество эпох
)

# Визуализация результатов обучения
acc = history.history['accuracy']  # Точность на тренировочном наборе
val_acc = history.history['val_accuracy']  # Точность на валидационном наборе

loss = history.history['loss']  # Потери на тренировочном наборе
val_loss = history.history['val_loss']  # Потери на валидационном наборе

epochs_range = range(epochs)  # Диапазон эпох

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Точность обучения')
plt.plot(epochs_range, val_acc, label='Точность валидации')
plt.legend(loc='lower right')
plt.title('Точность обучения и валидации')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Потери обучения')
plt.plot(epochs_range, val_loss, label='Потери валидации')
plt.legend(loc='upper right')
plt.title('Потери обучения и валидации')
plt.show()

# Сохранение модели
model.save('model/image_classification_model.h5')
