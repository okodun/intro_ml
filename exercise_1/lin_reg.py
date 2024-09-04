import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

# define x and y points
x_points = []
y_points = []


def on_click(event):

    # get left click
    if event.button is MouseButton.LEFT:
        # save coordinates
        x_points.append(event.xdata)
        y_points.append(event.ydata)

        # plot scatter
        plt.plot(event.xdata, event.ydata, "kx")
        plt.show()

    # get right click
    elif event.button is MouseButton.RIGHT and len(x_points) > 1:
        plt.clf()
        plt.plot(x_points, y_points, "kx")
        a, b = my_linfit(np.array(x_points), np.array(y_points))
        xp = np.arange(-2, 5, 0.1)
        plt.plot(xp, a * xp + b, "r-")
        print(f"My_fit: a={a} and b={b}")
        plt.show()


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
    plt.plot([], [], "kx")
    plt.connect("button_press_event", on_click)
    plt.show()
