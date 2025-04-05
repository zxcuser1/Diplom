import os
import warnings

import numpy as np
from skimage import io, color, filters, measure, morphology


def calculate_contour_area(contour):
    """Вычисляет площадь контура используя метод полигонального замыкания"""
    x = contour[:, 1]
    y = contour[:, 0]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# Функция для обработки каждого изображения
def process_image(image_path, output_folder):
    # Загрузка изображения
    image = io.imread(image_path)
    print(f"Загружено изображение: {image_path}, форма: {image.shape}")

    # Проверка и корректировка формы изображения
    if image.ndim > 3:
        # Сведение изображения к 3 измерениям, выбирая первый элемент по дополнительным осям
        image = image[0]
        while image.ndim > 3:
            image = image[0]

    # Проверяем, если изображение RGBA, конвертируем его в RGB
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]  # отбрасываем альфа-канал
        print(f"Изображение конвертировано из RGBA в RGB: {image.shape}")

    # Проверяем, если изображение RGB, конвертируем его в градации серого
    if image.ndim == 3 and image.shape[-1] == 3:
        gray = color.rgb2gray(image)
        print(f"Изображение конвертировано из RGB в градации серого: {gray.shape}")
    elif image.ndim == 2:
        # Изображение уже в градациях серого
        gray = image
        print(f"Изображение уже в градациях серого: {image.shape}")
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Применение размытия для уменьшения шума
    blurred = filters.gaussian(gray, sigma=2)

    # Применение адаптивной пороговой сегментации
    binary_image = filters.threshold_otsu(blurred)
    binary_image = blurred < binary_image

    # Морфологические операции для улучшения выделения
    binary_image = morphology.binary_closing(binary_image, morphology.disk(2))
    binary_image = morphology.remove_small_objects(binary_image, min_size=450)
    binary_image = morphology.remove_small_holes(binary_image, area_threshold=450)

    # Поиск контуров
    contours = measure.find_contours(binary_image, 0.8)

    # Создание пустой маски для графика
    mask = np.zeros_like(gray, dtype=np.uint8)

    # Вычисление площади каждого контура и сортировка по убыванию площади
    contours = sorted(contours, key=calculate_contour_area, reverse=True)

    if len(contours) == 0:
        raise ValueError('low contours')

    # Условие для удаления рамок и мелких контуров
    num_contours_to_keep = min(5, len(contours))  # Оставляем максимум 5 контуров
    for i in range(num_contours_to_keep):
        contour = contours[i]
        contour_area = calculate_contour_area(contour)
        if contour_area > 1000:
            rr, cc = contour[:, 0].astype(int), contour[:, 1].astype(int)
            mask[rr, cc] = 255

    # Сохранение маски с подавлением предупреждений о низкой контрастности
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        io.imsave(output_path, mask.astype(np.uint8))


# Путь к папке с исходными изображениями
input_folder = 'final_dataset_1'
# Путь к папке для сохранения результатов
output_folder = 'final_processing_dataset_1'

# Создаем папку для сохранения результатов, если её нет
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Обработка каждой подпапки внутри input_folder
for root, dirs, _ in os.walk(input_folder):
    for directory in dirs:
        # Полный путь к текущей подпапке
        current_folder = os.path.join(root, directory)

        # Проверяем, является ли текущий элемент подпапкой
        if os.path.isdir(current_folder):
            # Создаем подпапку в processed_dataset, если её нет
            output_subfolder = os.path.join(output_folder, directory)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            # Обработка каждого изображения в текущей подпапке
            for _, _, files in os.walk(current_folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_path = os.path.join(current_folder, file)
                        process_image(image_path, output_subfolder)

print("Обработка завершена.")
