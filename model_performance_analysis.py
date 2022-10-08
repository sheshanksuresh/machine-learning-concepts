import numpy as np
import matplotlib.pyplot as plt

x_int = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
X_train = np.linspace(0.0, 1.0, 10)  # training set
X_valid = np.linspace(0.0, 1.0, 100)  # validation set
np.random.seed(2679)
t_train = np.sin(4 * np.pi * X_train) + 0.3 * np.random.randn(10)
t_valid = np.sin(4 * np.pi * X_valid) + 0.3 * np.random.randn(100)

print("The training set is: ", X_train)
print("The validation set is: ", X_valid)
input("Press any key to continue...")

plt.scatter(x_int, X_train)
plt.title("Training and Validation Set")
plt.show()
