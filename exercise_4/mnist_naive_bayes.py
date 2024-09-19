import sys
import numpy as np
import tensorflow as tf


def class_acc(pred, gt):
    """returns accuracy"""

    assert pred.shape[0] == gt.shape[0]
    return (pred - gt == 0).sum() / pred.shape[0]


class NaiveBayesClassifier:
    """implementation of naive Bayes' classifier for MNIST"""

    def __init__(self):
        """creates a new NaiveBayesClassifier instance"""

        self.MEANS = None
        self.VARIANCES = None

    def fit(self, x, y):
        """fits the data by calculating mean and variance vectors"""

        # get mean and variance vectors by class
        means = []
        variances = []
        for i in list(set(y)):

            # get mean by dimension an append
            mean = np.mean(x[y == i], axis=0)
            means.append(mean)

            # calculate variance by dimension and append
            squared_loss = np.sum((x[y == i] - mean) ** 2, axis=0)
            variance = squared_loss / x[y == i].shape[0]
            variances.append(variance)

        # convert vectors to arrays
        self.MEANS = np.array(means)
        self.VARIANCES = np.array(variances)

    def predict(self, x):
        """predicts the class labels of x"""

        # initialize predictions
        pred = np.zeros(x.shape[0])

        # define constant term
        const_term = -0.5 * np.log(2 * np.pi)

        # calculate likelihoods
        for k in range(x.shape[0]):
            likelihoods = np.zeros(self.MEANS.shape[0])

            for i in range(self.MEANS.shape[0]):
                log_term = -0.5 * np.log(self.VARIANCES[i])
                square_term = -((x[k] - self.MEANS[i]) ** 2) / (2 * self.VARIANCES[i])
                likelihoods[i] = np.sum(const_term + log_term + square_term)

            pred[k] = np.argmax(likelihoods)

        return pred


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
    nbc = NaiveBayesClassifier()
    nbc.fit(x_train_reshaped, y_train)
    pred = nbc.predict(x_test_reshaped)

    # print prediction
    print(f"Classification accuracy is {round(class_acc(pred, y_test) * 100, 2)}%")
