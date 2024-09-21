import sys
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal


def class_acc(pred, gt):
    """returns accuracy"""

    assert pred.shape[0] == gt.shape[0]
    return (pred - gt == 0).sum() / pred.shape[0]


class FullBayesClassifier:
    def __init__(self):
        self.MEANS = None
        self.COVARIANCES = None

    def fit(self, x, y):
        means = []
        covariances = []

        # calculate mean vector and covariance matrix for each class
        for i in np.unique(y):

            means.append(np.mean(x[y == i], axis=0))
            covariances.append(np.cov(x[y == i], rowvar=False))

        self.MEANS = np.array(means)
        self.COVARIANCES = np.array(covariances)

    def predict(self, x):
        classes = self.MEANS.shape[0]
        likelihoods = np.zeros((x.shape[0], classes))
        for k in range(classes):
            likelihoods[:, k] = multivariate_normal.logpdf(
                x, mean=fbc.MEANS[k], cov=fbc.COVARIANCES[k]
            )
        return np.argmax(likelihoods, axis=1)


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

    # add noise
    x_train = x_train + np.random.normal(loc=0.0, scale=10.0, size=x_train.shape)

    # reshape matrices
    x_train_reshaped = np.reshape(x_train, (x_train.shape[0], -1))
    x_test_reshaped = np.reshape(x_test, (x_test.shape[0], -1))

    # classify test data
    fbc = FullBayesClassifier()
    fbc.fit(x_train_reshaped, y_train)

    # predict
    pred = fbc.predict(x_test_reshaped)

    # print prediction
    print(f"Classification accuracy is {round(class_acc(pred, y_test) * 100, 2)}%")
