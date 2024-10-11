import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.dat", encoding="UTF-8")
survival_status = data[:, 1]
gender = data[:, 2]
male_survivors = len(np.where((survival_status == 1) & (gender == 0))[0])
male_casualties = len(np.where((survival_status == 0) & (gender == 0))[0])
female_survivors = len(np.where((survival_status == 1) & (gender == 1))[0])
female_casualties = len(np.where((survival_status == 0) & (gender == 1))[0])

print(
    f"Out of {gender[gender==0].shape[0]} male passengers, {male_survivors} survived and {male_casualties} died."
)
print(f"(Survival rate: {(male_survivors/gender[gender==0].shape[0])*100:.2f}%)")
print(
    f"Out of {gender[gender==1].shape[0]} female passengers, {female_survivors} survived and {female_casualties} died."
)
print(f"(Survival rate: {(female_survivors/gender[gender==1].shape[0])*100:.2f}%)")
