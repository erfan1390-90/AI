import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def train_and_save():
    print("در حال بارگیری داده‌های MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    print("در حال ساخت مدل...")
    model = keras.Sequential([
        layers.InputLayer(shape=(28, 28, 1), name="input_layer"),
        layers.Rescaling(scale=1.0 / 255.0, name="rescaling"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation='relu', name="dense"),
        layers.Dense(10, activation='softmax', name="dense_1")
    ], name="sequential")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    print("شروع آموزش مدل...")
    model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(x_test, y_test)
    )

    model.save("mnist_model.keras")
    print("مدل با موفقیت در فایل mnist_model.keras ذخیره شد.")

if __name__ == "__main__":
    train_and_save()
