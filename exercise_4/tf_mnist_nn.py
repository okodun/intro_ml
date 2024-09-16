import sys
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier


def class_acc(pred, gt):
    """returns accuracy"""

    assert pred.shape[0] == gt.shape[0]
    return (pred - gt == 0).sum() / pred.shape[0]


if __name__ == "__main__":

    # parse cmdline arguments
    if len(sys.argv) != 2 or sys.argv[1].lower() not in ["original", "fashion"]:
        err_msg = """
        Invalid command line argument. 
        Usage: python tf_mnist_nn.py <original|fashion>
        """
        raise ValueError(err_msg)
    elif sys.argv[1].lower() == "original":
        mnist = tf.keras.datasets.mnist
    elif sys.argv[1].lower() == "fashion":
        mnist = tf.keras.datasets.fashion_mnist

    # load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape matrices
    x_train_reshaped = np.reshape(
        x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[1])
    )
    x_test_reshaped = np.reshape(
        x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[1])
    )

    # fit model
    knn = KNeighborsClassifier()
    knn.fit(x_train_reshaped, y_train)

    # predict model
    pred = knn.predict(x_test_reshaped)

    # print prediction
    print(f"Classification accuracy is {round(class_acc(pred, y_test) * 100, 2)}%")
