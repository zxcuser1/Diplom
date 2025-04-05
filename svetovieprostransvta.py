import cv2
import matplotlib.pyplot as plt
import numpy as np

# Загрузка изображения
graph = cv2.imread('1.png')

# Преобразование изображения из BGR в RGB
graph = cv2.cvtColor(graph, cv2.COLOR_BGR2RGB)

# Преобразование изображения из RGB в HSV
hsv_graph = cv2.cvtColor(graph, cv2.COLOR_RGB2HSV)

# Определение диапазона черного цвета для маскирования
black = (0, 0, 0)
white = (0, 0, 125)

# Создание маски, выделяющей черные и белые области в изображении
mask = cv2.inRange(hsv_graph, black, white)

# Применение маски к изображению для выделения нужных областей
result = cv2.bitwise_and(graph, graph, mask=mask)

# Применение гауссовского размытия к результату для уменьшения шума
blur = cv2.GaussianBlur(result, (1, 1), 0)

# Отображение результата с помощью matplotlib
plt.imshow(blur)
plt.axis('off')  # Отключение осей для чистоты изображения
plt.show()
