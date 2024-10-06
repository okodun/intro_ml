import numpy as np
import matplotlib.pyplot as plt


def generate_data_2D():
    """generates two normally distributed clusters of data"""

    # define constants
    # np.random.seed(42)
    num_points_per_cluster = 50
    cluster_centers = np.array([[0.1, 0.1], [0.6, 0.6]])
    cluster_std_dev = 0.1

    # define clusters
    cluster_1 = np.random.normal(
        loc=cluster_centers[0], scale=cluster_std_dev, size=(num_points_per_cluster, 2)
    )
    cluster_2 = np.random.normal(
        loc=cluster_centers[1], scale=cluster_std_dev, size=(num_points_per_cluster, 2)
    )

    # append data and return
    return np.vstack((cluster_1, cluster_2))


def generate_data_3D(n=1000, std_dev=1.0, seed=None):
    """generates four normally distributed clusters of 3D data"""

    # set seed if needed
    if seed is not None:
        np.random.seed(seed)

    # define the cluster centers in 3D space
    centers = np.array(
        [
            [0.8, 0.8, 0.8],  # Center of cluster 1
            [-0.8, -0.8, -0.8],  # Center of cluster 2
            [0.8, -0.8, 0.8],  # Center of cluster 3
            [-0.8, 0.8, -0.8],  # Center of cluster 4
        ]
    )

    # define number of samples per cluster
    samples_per_cluster = n // len(centers)

    # generate data points for each cluster and assign a class label
    X = []
    y = []
    for i, center in enumerate(centers):
        cluster_points = np.random.randn(samples_per_cluster, 3) * std_dev + center
        X.append(cluster_points)
        y.append(np.full(samples_per_cluster, i))

    # create arrays
    X = np.vstack(X)
    y = np.hstack(y)

    return X, y


class SOM:
    """class for Self-Organizing Map"""

    def __init__(self, neurons=10, dim=2, lr=0.5, epochs=30):
        """creates a new SOM instance"""

        self.LEARNING_RATE = lr
        self.NEURONS = np.random.rand(neurons, dim)
        self.EPOCHS = epochs

    def fit(self, x):
        """fits the data in x and calculates a Self-Organizing Map"""

        # train for predefined epochs
        for _ in range(self.EPOCHS):

            # shuffle order of datapoints
            np.random.shuffle(x)

            # find closest neurons for all datapoints
            for datapoint in x:

                # compute distance between neurons and datapoint anf find index of closest neuron
                distances = np.sqrt(np.sum((self.NEURONS - datapoint) ** 2, axis=1))
                idx = np.argmin(distances)

                # update closest neuron
                self.NEURONS[idx] += self.LEARNING_RATE * (
                    datapoint - self.NEURONS[idx]
                )

                # update left neighbor (with update rate of 0.5)
                if idx > 0:
                    self.NEURONS[idx - 1] += (
                        self.LEARNING_RATE * 0.5 * (datapoint - self.NEURONS[idx - 1])
                    )

                # update right neighbor (with update rate of 0.5)
                if idx < self.NEURONS.shape[0] - 1:
                    self.NEURONS[idx + 1] += (
                        self.LEARNING_RATE * 0.5 * (datapoint - self.NEURONS[idx + 1])
                    )


if __name__ == "__main__":

    # create 3D data
    X, y = generate_data_3D(n=1000, std_dev=0.2, seed=13)

    # create plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D clusters and their representation as Self-Organizing Map")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    # plot data
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
    # uncomment the following line to get rid of classes
    # scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2])

    # calculate SOM and plot
    som = SOM(dim=3, neurons=100, lr=0.2, epochs=30)
    som.fit(X)
    ax.plot(som.NEURONS[:, 0], som.NEURONS[:, 1], som.NEURONS[:, 2], color="red")
    ax.plot(som.NEURONS[:, 0], som.NEURONS[:, 1], color="orange")

    plt.show()
