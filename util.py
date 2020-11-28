################################################################################
# Util that immplements miscelaneous tasks.
#
# Author: Po-Kai Chang
################################################################################
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np

def split_train_test(X, y, test_ratio=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=1234, shuffle=True)

    # Summarize each class' distribution
    unique, counts = np.unique(y_test, return_counts=True)
    print("Summarize each class' distribution", dict(zip(unique, counts)))
    return X_train, X_test, y_train, y_test

def test_split_train_test():
    (_, _), (x_data, y_data) = mnist.load_data()
    X_train, X_test, y_train, y_test = split_train_test(x_data, y_data)