import sys
import numpy as np
import tensorflow as tf


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

    # add noise
    x_train = x_train + np.random.normal(loc=0.0, scale=10.0, size=x_train.shape)

    # reshape matrices
    x_train_reshaped = np.reshape(x_train, (x_train.shape[0], -1))
    x_test_reshaped = np.reshape(x_test, (x_test.shape[0], -1))

    # get mean and variance vectors by class
    mean_vecs = []
    variance_vecs = []
    for i in list(set(y_train)):

        # get mean by dimension an append
        mean = np.mean(x_train_reshaped[y_train == i], axis=0)
        mean_vecs.append(mean)

        # calculate variance by dimension and append
        mean_vec = np.sum((x_train_reshaped[y_train == i] - mean) ** 2, axis=0)
        variance = mean_vec / x_train_reshaped[y_train == i].shape[0]
        variance_vecs.append(variance)

    # convert vectors to arrays
    mean_vecs = np.array(mean_vecs)
    variance_vecs = np.array(variance_vecs)

    pred = np.zeros(y_test.shape[0])
    for k in range(y_test.shape[0]):

        # calculate likelihoods
        const = -0.5 * np.log(2 * np.pi)
        probabilities = np.zeros(mean_vecs.shape[0])

        for i in range(mean_vecs.shape[0]):
            t2 = -0.5 * np.log(variance_vecs[i])
            # t3 = -((x_test_reshaped[k] - mean_vecs[i]) ** 2) / (2 * variance_vecs[i])
            t3 = -((x_test_reshaped[k] - mean_vecs[i]) ** 2) / (2 * variance_vecs[i])
            probabilities[i] = np.sum(const + t2 + t3)

        pred[k] = np.argmax(probabilities)

    # print prediction
    print(f"Classification accuracy is {round(class_acc(pred, y_test) * 100, 2)}%")
