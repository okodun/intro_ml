import sys
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as mn


def class_acc(pred, gt):
    """returns accuracy"""

    assert pred.shape[0] == gt.shape[0]
    return (pred - gt == 0).sum() / pred.shape[0]


class FullBayesClassifier:
    """implementation of Bayes' classifier for MNIST"""

    def __init__(self):
        """creates a new FullBayesClassifier instance"""

        self.MEANS = None
        self.COVARIANCES = None

    def fit(self, x, y, current_noise):
        """fits the data by calculating mean vectors and covariance matrices"""

        means = []
        covariances = []

        # calculate mean vector and covariance matrix for each class
        for i in np.unique(y):

            # get all samples belonging to current class
            m = x[y == i]

            # calculate mean vector for class
            mean = np.sum(m, axis=0) / m.shape[0]
            means.append(mean)

            # calculate covariance matrix for class
            covariance = ((m - mean).T @ (m - mean)) / (m.shape[0] - 1)
            covariances.append(covariance)

        # convert results to numpy arrays and save them
        self.MEANS = np.array(means)
        self.COVARIANCES = np.array(covariances)

        # report rank values
        ok = True
        for i in range(self.COVARIANCES.shape[0]):
            rank = np.linalg.matrix_rank(self.COVARIANCES[i])
            if rank != self.MEANS.shape[1]:
                ok = False
                print(
                    f"Covariance matrix for class {i} contains only {rank} instead of {self.MEANS.shape[1]} dimensions."
                )
        if ok:
            print(
                f"No dimensions vanished during covariance matrix computation at current noise level {current_noise}. Shape is {self.MEANS.shape[1]}x{self.MEANS.shape[1]}."
            )

    def predict(self, x):
        """predicts the class labels of x"""

        # define class labels and create array for likelihoods
        classes = self.MEANS.shape[0]
        likelihoods = np.zeros((x.shape[0], classes))

        # calculate likelihoods
        for k in range(classes):

            # compute likelihood for all samples per class
            likelihoods[:, k] = mn.logpdf(x, mean=fbc.MEANS[k], cov=fbc.COVARIANCES[k])

        # return index of maximum likelihood per sample
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

    # determine if adding noise improves prediction accuracy
    accuracy = 0
    current_noise = 0
    noise_levels = [0.1, 1.0, 10.0]

    for i in range(len(noise_levels)):

        # add noise and save current noise
        x_train = x_train + np.random.normal(
            loc=0.0, scale=noise_levels[i], size=x_train.shape
        )
        current_noise = noise_levels[i]

        # reshape matrices
        x_train_reshaped = np.reshape(x_train, (x_train.shape[0], -1))
        x_test_reshaped = np.reshape(x_test, (x_test.shape[0], -1))

        # classify test data
        fbc = FullBayesClassifier()
        fbc.fit(x_train_reshaped, y_train, current_noise)
        pred = fbc.predict(x_test_reshaped)

        # determine accuracy
        acc = round(class_acc(pred, y_test) * 100, 2)
        if i > 0 and accuracy > acc:
            break
        accuracy = acc

    # print prediction
    print(f"\nClassification accuracy is {accuracy}%")

    # print used noise if improves
    if current_noise != 0.1:
        print(
            f"""Prediction accuracy is highest with noise scale {current_noise}.\nTested with noise scale 0.1, 1.0, and 10.0."""
        )
