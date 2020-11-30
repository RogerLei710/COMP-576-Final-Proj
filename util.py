################################################################################
# Util that immplements miscelaneous tasks.
#
# Author: Po-Kai Chang
################################################################################
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def split_train_test(X, y, test_ratio=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=1234, shuffle=True)

    # Summarize each class' distribution
    unique, counts = np.unique(y_test, return_counts=True)
    print("(Test) Summarize each class' distribution", dict(zip(unique, counts)))
    unique, counts = np.unique(y_train, return_counts=True)
    print("(Train) Summarize each class' distribution", dict(zip(unique, counts)))
    return X_train, X_test, y_train, y_test


# plotting helper function
def plot_hist(hist):
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def test_split_train_test():
    (_, _), (x_data, y_data) = mnist.load_data()
    X_train, X_test, y_train, y_test = split_train_test(x_data, y_data)
