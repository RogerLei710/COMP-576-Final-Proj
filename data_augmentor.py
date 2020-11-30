################################################################################
# Data augmentators.
#
# It should correspond to the augmentor(s) used in the paper "Practical Black-Box Attacks against Machine Learning".
#
# Author: Po-Kai Chang, Roger
################################################################################
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.datasets import mnist


def Jacobian(model, data):
    """Jacobian data augmentation method.
    It returns the augmented (newly created) data.
    - model: some model that we can perform dFi / dxj on softmax of jth class w.r.t. jth input entry
    - data: numpy array of shape(N, M), where N is the size of the dataset, M is the feature num
    """
    _data = tf.convert_to_tensor(data, dtype=tf.float32)  # Convert data to tensor
    with tf.GradientTape(persistent=True) as tape:
        x = _data
        tape.watch(x)
        y = model(x)

    batch_jacobian = tape.batch_jacobian(y, x)
    return batch_jacobian


def cifar10_data_generator(x_train):
    """
    set up image augmentation for cifar 10
    :param x_train: x_train data
    :return: data generator
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
        # zoom_range=0.3
    )
    datagen.fit(x_train)
    return datagen


def test_Jacobian():
    """Test with pretrained model + its corresponding dataset
    """
    # Load model
    model = keras.models.load_model('saved_models/mnist_target_model')
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(lr=1e-3),
        metrics=["accuracy"],
    )

    # Get test data
    img_rows, img_cols, channels = 28, 28, 1
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape((-1, img_rows, img_cols, channels))

    # Mnotor on input (x) to get jacobian
    batch_jacobian = Jacobian(model, x_test)
    print(batch_jacobian)
