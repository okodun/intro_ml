import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.dat", encoding="UTF-8")
survival_status = data[:, 1]
gender = data[:, 2]
males_surv = data[:, 4][np.where((survival_status == 1) & (gender == 0))[0]]
males_died = data[:, 4][np.where((survival_status == 0) & (gender == 0))[0]]
females_surv = data[:, 4][np.where((survival_status == 1) & (gender == 1))[0]]
females_died = data[:, 4][np.where((survival_status == 0) & (gender == 1))[0]]

plt.subplot(1, 2, 1)
plt.hist(males_surv, bins=10, color="green", alpha=0.5)
plt.hist(males_died, bins=10, color="red", alpha=0.5)
plt.title("Ticket Fare Distribution (Males)")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.subplot(1, 2, 2)
plt.hist(females_surv, bins=10, color="green", alpha=0.5)
plt.hist(females_died, bins=10, color="red", alpha=0.5)
plt.title("Ticket Fare Distribution (Females)")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")


plt.show()
