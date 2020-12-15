################################################################################
# Demo script that immplements white box attack on trained CNN on Mnist dataset.
#
# It loads stored model from saved_models/cifar10, using its gradient directly to
# generate a perturbed image for each input.
#
# Usage: python3 cifar10_attack.py
# 
# Author: Roger
################################################################################
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# this part will make sure the GPU memory will not be drained totally and will prevent from failing.
# https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import matplotlib.pyplot as plt
import drawHeatMap as dhm
from sklearn.metrics import classification_report, confusion_matrix

(_, _), (x_test, y_test) = cifar10.load_data()
# Ref: https://www.cs.toronto.edu/~kriz/cifar.html
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
img_rows, img_cols, channels = 32, 32, 3
num_classes = 10

x_test = x_test.astype("float32") / 255.0

x_test = x_test.reshape((-1, img_rows, img_cols, channels))

y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)

def get_substitute_model(loc='saved_models/cifar10_substitute_model/datagen'):
    print('Get substitute model from {}'.format(loc))
    substitute_model = keras.models.load_model(loc)
    substitute_model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(lr=1e-3),
        metrics=["accuracy"],
    )
    return substitute_model


target_model = keras.models.load_model('saved_models/cifar10_target_model')

target_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=1e-3),
    metrics=["accuracy"],
)


def make_adversarial(image, label, epsilon=0.2):
    perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), label).numpy()
    adversarial = np.clip(image + perturbations * epsilon, 0.0, 1.0)
    return adversarial


def adversarial_pattern(image, label):
    image = tf.cast(image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = substitute_model(image)
        loss = tf.keras.losses.MSE(label, prediction)
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    return signed_grad


def generate_adversarials(batch_size, epslion=0.2):
    while True:
        x = []
        y = []
        for batch in range(batch_size):
            # N = random.randint(0, y_test.shape[0] - 1)
            image = x_test[batch]
            label = y_test_one_hot[batch]

            adversarial = make_adversarial(image, label, epslion)

            x.append(adversarial)
            y.append(y_test[batch])

        x = np.asarray(x).reshape((batch_size, img_rows, img_cols, channels))
        y = np.asarray(y)

        yield x, y


# plot first x misclassification pictures
def plot_misclassifications(miscount, idx, epslion=0.2):
    base = miscount
    while miscount > 0:
        image = x_test[idx]
        image = image.reshape((1, img_rows, img_cols, channels))
        image_label = y_test_one_hot[idx]
        adversarial = make_adversarial(image, image_label, epslion)

        plt.subplot(base, 2, 2 * (base - miscount) + 1)
        plt.title("original")
        plt.imshow(image.reshape((img_rows, img_cols, channels)))

        plt.subplot(base, 2, 2 * (base - miscount) + 2)
        plt.title("adversarial")
        if labels[substitute_model.predict(image).argmax()] != labels[substitute_model.predict(adversarial).argmax()]:
            miscount -= 1
            label_orig, label_classified = labels[substitute_model.predict(image).argmax()], labels[
                substitute_model.predict(adversarial).argmax()]
            if channels == 1:
                adversarial_img = adversarial.reshape((img_rows, img_cols))
                plt.imshow(adversarial_img)
            else:
                plt.imshow(adversarial.reshape((img_rows, img_cols, channels)))
            plt.show()
            print('The original image is classified as {}'.format(label_orig))
            print('The adversarial image is classified as {}'.format(label_classified))
        idx += 1




# Run experiment with target model and substitute model (varies)
def run_experiment(substitute_model_loc='saved_models/cifar10_target_model', plot=False, epslion=0.1):
    global substitute_model
    substitute_model = get_substitute_model(loc=substitute_model_loc)
    adversarial_num = 1000
    x_adversarial_test, y_adversarial_test = next(generate_adversarials(batch_size=adversarial_num, epslion=epslion))
    # print("Accuracy of Substitute model on regular images:", substitute_model.evaluate(x=x_test, y=y_test, verbose=0))
    # print("Accuracy of Target model on regular images:", target_model.evaluate(x=x_test, y=y_test, verbose=0))


    # print("Accuracy of Substitute model on adversarial images:",
    #       substitute_model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))
    print("Accuracy of Target model on adversarial images:",
          target_model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))

    y_sub = substitute_model.predict(x_test).argmax(axis=1)
    y_target = target_model.predict(x_test).argmax(axis=1)
    # print(y_.shape)
    # print(y_test.shape)

    ################Confusion Matrix#################
    print('Substitute Confusion Matrix')
    print(confusion_matrix(y_sub, y_test))
    matrix = confusion_matrix(y_sub, y_test)
    # y_axis_title = [str(x) for x in range(10)]
    # x_axis_title = [str(x) for x in range(10)]
    y_axis_title = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    x_axis_title = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    fig, ax = plt.subplots()
    im, cbar = dhm.heatmap(matrix, y_axis_title, x_axis_title, ax=ax,
                       cmap="Blues", cbarlabel="Labeled Count", title="Substitute Model Confusion Matrix")
    texts = dhm.annotate_heatmap(im, valfmt="{x:d}")
    fig.tight_layout()
    plt.show()

    print('Target Confusion Matrix')
    print(confusion_matrix(y_target, y_test))
    matrix = confusion_matrix(y_target, y_test)
    fig, ax = plt.subplots()
    im, cbar = dhm.heatmap(matrix, y_axis_title, x_axis_title, ax=ax,
                       cmap="Greens", cbarlabel="Labeled Count", title="Target Model Confusion Matrix")
    texts = dhm.annotate_heatmap(im, valfmt="{x:d}")
    fig.tight_layout()
    plt.show()
    #################################################

    if plot:
        plot_misclassifications(miscount=5, idx=0, epslion=epslion)



if __name__ == "__main__":
    model_loc_substitute = 'saved_models/cifar10_substitute_model/datagen'
    for e in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        run_experiment(substitute_model_loc=model_loc_substitute, plot=False, epslion=e)