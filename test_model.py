import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Отключение предупреждений TensorFlow
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Путь к данным
test_gr = Path('test')

batch_size = 32
img_width = 256
img_height = 256

# Загрузка тестового набора данных
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_gr,  # Путь к тестовому набору данных
    image_size=(img_height, img_width),  # Размер изображения
    batch_size=batch_size  # Размер пакета
)

# Кэширование и предвыборка данных для ускорения обработки
AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Загрузка обученной модели
model = tf.keras.models.load_model('model/image_classification_model.h5')

# Инициализация списков для истинных и предсказанных меток
y_true = []
y_pred = []

# Прогнозирование на тестовом наборе данных
for images, labels in test_ds:
    preds = model.predict(images)  # Предсказание меток для изображений
    y_true.extend(labels.numpy())  # Добавление истинных меток
    y_pred.extend(np.argmax(preds, axis=1))  # Добавление предсказанных меток

# Преобразование списков в numpy массивы
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Вычисление метрик
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Предсказание вероятностей для вычисления ROC AUC
y_pred_prob = model.predict(test_ds)
roc_auc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr')

# Вывод метрик
print(f"Validation Accuracy: {accuracy}")
print(f"Validation Precision: {precision}")
print(f"Validation Recall: {recall}")
print(f"Validation F1 Score: {f1}")
print(f"Validation ROC AUC: {roc_auc}")
