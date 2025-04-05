import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras import layers, models
from pathlib import Path

# Создание генератора изображений
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 # 20% данных будут задействованы для валидации
)

# Создание обучающего набора данных с разделением на обучающую и валидационную выборки
train_generator = train_datagen.flow_from_directory(
    directory=Path('dataset'),  # Путь до датасета
    target_size=(128, 128), # Разммер изображения
    batch_size=32,
    color_mode="grayscale", # Цвет изображения
    class_mode='categorical',
    subset='training',  # обучающая выборка
    seed=50
)

# Создание валидационного набора данных
val_generator = train_datagen.flow_from_directory(
    directory=Path('dataset'),  # Путь до датасета
    target_size=(128, 128), # Разммер изображения
    batch_size=32,
    color_mode="grayscale", # Цвет изображения
    class_mode='categorical',
    subset='validation',    # валидационная выборка
    seed=50
)

# Создание нейронной сети
model = models.Sequential()
# Первый сверточный слой
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Второй сверточный слой
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Третий сверточный слой
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.TimeDistributed(layers.Flatten())) # Преобразует выходы из сверточных слоев в 3-х мерный формат,необходимый для LTSM

model.add(layers.LSTM(128)) # Полносвязный LSTM слой
model.add(layers.Dropout(0.5))  # отсев
model.add(layers.Dense(9, activation='softmax')) # Выходной слой с функцией активации softmax для многоклассовой классификации

# Компиляция модели
model.compile(optimizer='adam',# Оптимизатор Adam
              loss='categorical_crossentropy', # Функция потерь для многоклассовой классификации
              metrics=['accuracy'])# Метрика точности

# Вывод структуры модели
model.summary()

# Обучение модели
history = model.fit(
    train_generator, # Обучающая выборка
    validation_data=val_generator, # Валидационная выборка
    epochs=10  # Количество эпох
)

# Визуализация результатов обучения
plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')   # График точности на обучающей выборке
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе') # График точности на валидационной выборке
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

plt.plot(history.history['loss'],
         label='Ошибка на обучающем наборе')    # График потерь на обучающей выборке
plt.plot(history.history['val_loss'],
         label='Ошибка на проверочном наборе')  # График потерь на валидационной выборке
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.show()

# сохранение модели
model.save('LTSM_model.h5')
