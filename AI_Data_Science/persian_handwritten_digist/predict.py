import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

def predict_digit(image_path, model_path="mnist_model.keras"):
    try:
        model = keras.models.load_model(model_path)
    except Exception as e:
        print(f"خطا در بارگذاری مدل: {e}")
        return

    try:
        img = Image.open(image_path).convert('L').resize((28, 28))
        img_array = np.array(img).reshape((1, 28, 28, 1))

        predictions = model.predict(img_array)
        predicted_digit = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

        print(f"--- نتایج پیش‌بینی ---")
        print(f"تصویر: {image_path}")
        print(f"عدد تشخیص داده شده: {predicted_digit}")
        print(f"درصد اطمینان: {confidence:.2f}%")
        return predicted_digit, confidence
    except Exception as e:
        print(f"خطا در پردازش تصویر: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "test.png"
    predict_digit(img_path)
