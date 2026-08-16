# 🧠 تشخیص ارقام دست‌نویس (MNIST) با Keras و TensorFlow

یک پروژه ساده، سبک و جامع برای تشخیص ارقام دست‌نویس با استفاده از مجموعه داده **MNIST** و فریم‌ورک **TensorFlow / Keras**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-V3-red.svg)

---

## 📌 فهرست مطالب
- [درباره پروژه](#-درباره-پروژه)
- [معماری و ساختار مدل](#-معماری-و-ساختار-مدل)
- [ساختار فایل‌های پروژه](#-ساختار-فایلهای-پروژه)
- [نصب و راه‌اندازی](#-نصب-و-راهاندازی)
- [نحوه استفاده](#-نحوه-استفاده)
  - [۱. آموزش مدل](#۱-آموزش-مدل)
  - [۲. پیش‌بینی روی تصاویر جدید](#۲-پیشبینی-روی-تصاویر-جدید)
- [نتایج و عملکرد](#-نتایج-و-عملکرد)

---

## 📖 درباره پروژه

این مخزن شامل پیاده‌سازی یک شبکه عصبی پیش‌سو (Feedforward Neural Network) است که روی دیتابیس معروف **MNIST** (شامل ۷۰,۰۰۰ تصویر خاکستری ۲۸×۲۸ از ارقام ۰ تا ۹) آموزش دیده است.

### ویژگی‌های کلیدی:
* انجام پیش‌پردازش ورودی‌ها (مقیاس‌بندی پیکسل‌ها و تبدیل ماتریس به بردار) به صورت لایه‌های داخلی در خود مدل.
* ذخیره‌سازی مدل با فرمت مدرن Keras V3 (`.keras`).
* اسکریپت ساده جهت بارگذاری مدل و پیش‌بینی روی تصاویر جدید.

---

## 🏗 معماری و ساختار مدل

مدل از نوع Sequential بوده و لایه‌های آن به شرح زیر تنظیم شده‌اند:

| شماره لایه | نام لایه | نوع (Type) | ابعاد ورودی / خروجی | جزئیات و پارامترها |
| :---: | :---: | :---: | :---: | :--- |
| **0** | `input_layer` | `InputLayer` | `(None, 28, 28, 1)` | دریافت تصاویر ۲۸×۲۸ تک‌کاناله |
| **1** | `rescaling` | `Rescaling` | `(None, 28, 28, 1)` | نرمال‌سازی مقادیر پیکسل‌ها از $[0, 255]$ به $[0, 1]$ |
| **2** | `flatten` | `Flatten` | `(None, 784)` | تبدیل ماتریس $28 \times 28$ به بردار ۱ بعدی ۷۸۴ تایی |
| **3** | `dense` | `Dense` | `(None, 128)` | لایه پنهان با **۱۲۸ نورون** و تابع فعال‌سازی **ReLU** |
| **4** | `dense_1` | `Dense` | `(None, 10)` | لایه خروجی با **۱۰ نورون** و تابع فعال‌سازی **Softmax** |

### تنظیمات آموزش (Compile Config):
* **بهینه‌ساز (Optimizer):** `Adam` با نرخ یادگیری $lr = 0.001$
* **تابع زیان (Loss Function):** `sparse_categorical_crossentropy`
* **معیار ارزیابی (Metrics):** `accuracy`

---

## 📁 ساختار فایل‌های پروژه

```text
.
├── train.py              # اسکریپت آموزش و ذخیره‌سازی مدل
├── predict.py            # اسکریپت پیش‌بینی روی تصاویر تست
├── mnist_model.keras     # فایل مدل ذخیره‌شده (فرمت Keras V3)
├── requirements.txt      # کتابخانه‌های مورد نیاز
└── README.md             # مستندات پروژه
```

---

## 🚀 نصب و راه‌اندازی

### پیش‌نیازها
از نصب بودن پایتون (نسخه ۳.۸ به بالا) روی سیستم خود مطمئن شوید.

### مراحل نصب

۱. ایجاد محیط مجازی (اختیاری اما پیشنهادی):
   ```bash
   python -m venv venv
   source venv/bin/activate  # در لینوکس و مک
   # venv\Scripts\activate  # در ویندوز
   ```

۲. نصب کتابخانه‌های مورد نیاز:
   ```bash
   pip install -r requirements.txt
   ```
   *(یا به صورت دستی: `pip install tensorflow numpy pillow matplotlib`)*

---

## 💻 نحوه استفاده

### ۱. آموزش مدل
برای آموزش مدل روی داده‌های MNIST و ذخیره آن، فایل `train.py` را اجرا کنید:

```bash
python train.py
```

### ۲. پیش‌بینی روی تصاویر جدید
برای تست مدل روی یک تصویر سفارشی (مثلاً `test.png`):

```python
import numpy as np
from PIL import Image
from tensorflow import keras

# بارگذاری مدل
model = keras.models.load_model("mnist_model.keras")

def predict_digit(image_path):
    # باز کردن تصویر، تبدیل به خاکستری و تغییر سایز به 28x28
    img = Image.open(image_path).convert('L').resize((28, 28))
    img_array = np.array(img).reshape((1, 28, 28, 1))
    
    # انجام پیش‌بینی
    predictions = model.predict(img_array)
    predicted_digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    return predicted_digit, confidence

# اجرای نمونه
digit, conf = predict_digit("test.png")
print(f"عدد پیش‌بینی شده: {digit} (میزان اطمینان: {conf:.2f}%)")
```

---

## 📊 نتایج و عملکرد

| معیار | دقت (Accuracy) |
| :--- | :---: |
| **دقت آموزش (Train Accuracy)** | حدود $98.5\%$ |
| **دقت تست (Test Accuracy)** | حدود $97.8\%$ |
