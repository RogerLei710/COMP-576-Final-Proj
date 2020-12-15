################################################################################
# Substitute CNN model trained with the help of a target model on Mnist dataset using tf2.
#
# It saves/overwrites model and its configs under saved_models/cifar10_substitute_model/.
#
# Usage: python3 cifar10_substitue_model.py
#
# Author: Roger
################################################################################
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

#https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow import keras
from tensorflow.keras import layers
import data_augmentor as da
import numpy as np
from util import split_train_test, plot_hist

target_model = keras.models.load_model('saved_models/cifar10_target_model')
(_, _), (x_data, y_data) = cifar10.load_data()
img_rows, img_cols, channels = 32, 32, 3
num_classes = 10

# 1. initial data collection
x_data = x_data.astype("float32") / 255.0
x_train, x_test, y_train, y_test = split_train_test(x_data, y_data, test_ratio=0.8)

x_train = x_train.reshape((-1, img_rows, img_cols, channels))
x_test = x_test.reshape((-1, img_rows, img_cols, channels))

y_train_prob = target_model.predict(x_train)
y_train = y_train_prob.argmax(axis=-1)


# 2. substitute model architecture
def create_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(img_rows, img_cols, channels)),
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax'),
        ]
    )
    return model


# 3. substitute training
def train_sub(model, x_train, y_train, x_test, y_test, epochs, lamda, aug_func='jacobian-alpha'):
    if aug_func == 'datagen':
        datagen = da.cifar10_data_generator(x_train)
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=256),
                            steps_per_epoch=len(x_train) / 256, epochs=50,
                            validation_data=(x_test, y_test))
    elif(aug_func in ['jacobian-alpha', 'jacobian-beta']):
        for iter in range(epochs):
            print("Train substitute network round {} / {} ...".format(iter, epochs))
            # train the ith dataset 10 epochs
            model.fit(x_train, y_train, batch_size=256, epochs=10, validation_data=(x_test, y_test))
            # data augmentation
            # batch_jacobian shape: (batch_size, num_classes, img_rows, img_cols, channels)
            batch_jacobian = da.Jacobian(model, x_train)
            # build indices list to get elements (batch_size, img_rows, img_cols, channels) from batch_jacobian
            indices = []
            for idx in range(batch_jacobian.shape[0]):
                indices.append([idx, y_train[idx]])
            # now we get the elements
            x_gradient = tf.gather_nd(batch_jacobian, indices)
            # x + lambda * sgn(JF(x)[O(x)])
            if aug_func == 'jacobian-alpha':
                x_delta = lamda * tf.sign(x_gradient)
            elif aug_func == 'jacobian-beta':
                normalized = tf.math.l2_normalize(x_gradient, axis=[1, 2], epsilon=1e-12)
                x_delta = lamda * normalized
            else:
                raise ValueError('Augmentation func {} not recognized'.format(aug_func))
            x_new_train = x_train + x_delta
            # Clip data, each pixel is only valid in [0.0, 1.0]
            x_new_train = np.clip(x_new_train, 0.0, 1.0)
            y_new_train_prob = target_model.predict(x_new_train)
            y_new_train = y_new_train_prob.argmax(axis=-1)
            x_train = tf.concat([x_train, x_new_train], 0)
            y_train = tf.concat([y_train, y_new_train], 0)
        # the augmented data from the last time in the loop needs to be trained
        print("Train substitute network round {} / {} ...".format(epochs, epochs))
        model.fit(x_train, y_train, batch_size=256, epochs=10, validation_data=(x_test, y_test))
    else:
        print("Wrong aug_func")

def run_experiment(lamda=0.001, aug_func='jacobian-alpha', save_model=True, epochs=4):
    model = create_model()
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(lr=1e-3),
        metrics=["accuracy"],
    )

    train_sub(model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epochs=epochs, lamda=lamda,
              aug_func=aug_func)
    print("(substitute model) Base accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))

    if save_model:
        loc = "saved_models/cifar10_substitute_model/{}".format(aug_func)
        print("Model saved under {}".format(loc))
        model.save(loc)


if __name__ == "__main__":
    run_experiment(lamda=0.1, aug_func='datagen', save_model=True, epochs=2)
