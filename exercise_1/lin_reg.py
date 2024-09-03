import numpy as np
import matplotlib.pyplot as plt


def my_linfit(x, y):
    # calculate a
    a1 = (x.dot(y)) - ((np.sum(x) * np.sum(y)) / x.shape[0])
    a2 = x.dot(x) - ((np.sum(x) * np.sum(x)) / x.shape[0])
    a = a1 / a2

    # calculate b
    b1 = np.sum(y) - (np.sum(x) * x.dot(y) / x.dot(x))
    b2 = x.shape[0] - (np.sum(x) * np.sum(x) / x.dot(x))
    b = b1 / b2

    return a, b


if __name__ == "__main__":
    x = np.random.uniform(-2, 5, 10)
    y = np.random.uniform(0, 3, 10)
    a, b = my_linfit(x, y)
    plt.plot(x, y, "kx")
    xp = np.arange(-2, 5, 0.1)
    plt.plot(xp, a * xp + b, "r-")
    print(f"My_fit: a={a} and b={b}")
    plt.show()
