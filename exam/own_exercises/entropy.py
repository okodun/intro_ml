import numpy as np


def entropy(x, y):
    """calculates entropy"""

    # check shape
    assert x.shape[0] == y.shape[0]

    # get entropy
    entropy = 0

    # get class entropy
    for k in np.unique(y):
        pk = x[y == k].shape[0] / x.shape[0]
        entropy -= pk * np.log2(pk)

    return entropy


def get_accuracy(pred, y):

    assert pred.shape[0] == y.shape[0]
    return (pred - y == 0).sum() / y.shape[0]


if __name__ == "__main__":
    x = np.random.rand(10, 2)
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    e = entropy(x, y)
    print(entropy(x, y))
    pred = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2, 2])
    print(get_accuracy(pred, y))
