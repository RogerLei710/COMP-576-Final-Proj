################################################################################
# Baseline CNN model on Mnist dataset using tf2.
#
# It saves/overwrites model and its configs under saved_models/cifar10_target_model/.
#
# Usage: python3 cifar10_target_model.py
#
# Author: Roger, Po-Kai Chang
################################################################################
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.datasets import cifar10

from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img_rows, img_cols, channels = 32, 32, 3
num_classes = 10

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((-1, img_rows, img_cols, channels))
x_test = x_test.reshape((-1, img_rows, img_cols, channels))


def create_model():
    # Ref: https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
    model = keras.Sequential(
        [
            keras.Input(shape=(img_rows, img_cols, channels)),
            layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ]
    )
    return model


def run_experiment(save_model=True):
    model = create_model()
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(lr=1e-3),
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, batch_size=256, epochs=50, validation_data=(x_test, y_test))
    print("Base accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))

    if save_model:
        model.save("saved_models/cifar10_target_model/")


if __name__ == "__main__":
    run_experiment()
