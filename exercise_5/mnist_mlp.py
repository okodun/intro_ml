import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def class_acc(pred, gt):
    """returns accuracy"""

    assert pred.shape[0] == gt.shape[0]
    return (pred - gt == 0).sum() / pred.shape[0]


# define parameters as constants
HIDDEN_LAYER = 784
LEARNING_RATE = 0.2
EPOCHS = 200

# load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape matrices
x_train_reshaped = np.reshape(x_train, (x_train.shape[0], -1))
x_test_reshaped = np.reshape(x_test, (x_test.shape[0], -1))

# get one-hot encoding for training labels
y_train_hot = np.zeros((y_train.size, y_train.max() + 1), dtype=int)
y_train_hot[np.arange(y_train.size), y_train] = 1

# create neural network
model = Sequential()

# add hidden layer with 784 neurons
model.add(Dense(HIDDEN_LAYER, input_dim=784, activation="sigmoid"))

# add output layer with 10 neurons
model.add(Dense(10, input_dim=HIDDEN_LAYER, activation="sigmoid"))

# compile for stochastic gradient descent
opt = keras.optimizers.SGD(learning_rate=LEARNING_RATE)
model.compile(optimizer=opt, loss="mse", metrics=["mse"])

# train
tr_hist = model.fit(x_train_reshaped, y_train_hot, epochs=EPOCHS, verbose=1)

# predict test dataset and print accuracy
pred = np.argmax(model.predict(x_train_reshaped), axis=1)
print(
    f"\nClassification accuracy for training data is {round(class_acc(pred, y_train) * 100, 2)}%"
)

# predict test dataset and print accuracy
pred = np.argmax(model.predict(x_test_reshaped), axis=1)
print(
    f"Classification accuracy for test data is {round(class_acc(pred, y_test) * 100, 2)}%"
)

# plot loss while learning
plt.plot(tr_hist.history["loss"])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

# 0.1 -> 30.22%
# 0.2 -> 37.54%
# 0.3 -> 51.70%
# 0.4 -> 30.05%
# 0.3 (100) -> 42.83%
# 0.2 (50n/30) -> 93.55%
