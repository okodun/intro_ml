import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.dat", encoding="UTF-8")
survived = data[:, 1].sum()
died = (data[:, 1] == 0).sum()

fsurvived = data[:, 4][data[:, 1] == 1]
fdied = data[:, 4][data[:, 1] == 0]

plt.figure(figsize=(10, 7))
plt.hist(fsurvived, color="green", alpha=0.5, label="survived accident")
plt.hist(fdied, color="red", alpha=0.5, label="died in accident")

plt.xlabel("Amount of paid fare")
plt.ylabel("Number of passangers")
plt.title("Fares paid by survived vs. dead passengers")
plt.legend()
plt.show()
