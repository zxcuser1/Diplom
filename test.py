from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

def pred(model, filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_mask = np.argmax(prediction, axis=-1)[0]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img[0])
    plt.title("Исходное изображение")

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap="jet")
    plt.title("Предсказанная маска")

    plt.show()


for i in range(2, 3):
    model = load_model(f'Unet256/final_model.h5')

    test_image_path = f'C:/Users/Danila/Desktop/g/21.png'

    pred(model, test_image_path)


