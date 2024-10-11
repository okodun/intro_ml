import numpy as np

t = np.loadtxt("data.dat", encoding="UTF-8")
survived = t[:, 1].sum()
died = (t[:, 1] == 0).sum()


out_string = f"Out of {int(survived+died)} passengers {int(survived)} ({round(survived/(survived+died)*100,2)}%) survived and {int(died)} ({round(died/(survived+died)*100,2)}%) died."
print(out_string)
