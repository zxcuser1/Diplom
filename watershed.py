import cv2
import numpy as np
from IPython.display import Image, display
from matplotlib import pyplot as plt

# Функция для отображения изображения
def imshow(img, ax=None):
    if ax is None:
        ret, encoded = cv2.imencode(".png", img)
        display(Image(encoded))
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')

# Чтение изображения
img = cv2.imread("image5_4.png")

# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Бинаризация изображения с помощью порогового значения Otsu
ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Создание структурного элемента и морфологическая операция открытия
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)

# Определение фона с помощью дилатации
sure_bg = cv2.dilate(bin_img, kernel, iterations=3)

# Преобразование расстояний для нахождения переднего плана
dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)

# Определение переднего плана с помощью порогового значения
ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
sure_fg = sure_fg.astype(np.uint8)

# Определение неизвестной области
unknown = cv2.subtract(sure_bg, sure_fg)

# Маркировка областей
ret, markers = cv2.connectedComponents(sure_fg)
markers += 1
markers[unknown == 255] = 0

# Применение алгоритма водораздела
markers = cv2.watershed(img, markers)

# Получение уникальных меток
labels = np.unique(markers)

# Выделение контуров объектов
coins = []
for label in labels[2:]:
    target = np.where(markers == label, 255, 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coins.append(contours[0])

# Отрисовка контуров на изображении
img = cv2.drawContours(img, coins, -1, color=(0, 23, 223), thickness=2)
imshow(img)
