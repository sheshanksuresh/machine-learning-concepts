import numpy as np
import matplotlib.pyplot as plt

def calculate_least_squares(feature_matrix, target_matrix):
    feature_matrix_transpose = np.transpose(feature_matrix)
    feature_multiply = np.matmul(feature_matrix, feature_matrix_transpose)
    featureT_multiply_target = np.matmul(feature_matrix_transpose, target_matrix)
    w_values = np.matmul(np.linalg.inv(feature_multiply), featureT_multiply_target)
    return w_values
X_train = np.linspace(0.0, 1.0, 10)  # training set
X_valid = np.linspace(0.0, 1.0, 100)  # validation set
np.random.seed(2679)
t_train = np.sin(4 * np.pi * X_train) + 0.3 * np.random.randn(10)
t_valid = np.sin(4 * np.pi * X_valid) + 0.3 * np.random.randn(100)

X_train_mat = np.reshape(X_train, (-1, 1))
X_train_mat = np.insert(X_train_mat,0,np.ones(10), axis=1)
X_valid_mat = np.reshape(X_valid, (-1, 1))
X_valid_mat = np.insert(X_valid_mat, 0, np.ones(100), axis=1)
t_train_mat = np.reshape(t_train,(-1, 1))

X_train_T = np.transpose(X_train_mat)
X_valid_T = np.transpose(X_valid_mat)

X_train_XT = np.matmul(X_train_T, X_train_mat)
X_valid_XT = np.matmul(X_valid_T, X_valid_mat)

t_train_XT = np.matmul(X_train_T, t_train_mat)

w_values = np.matmul(np.linalg.inv(X_train_XT), t_train_XT)

print(w_values)
print("**************************************************")
print(calculate_least_squares(X_train_mat, t_train_mat))
# print("The y matrix is: ", t_train_mat)
# print("The validation set is: ", X_valid)
# input("Press any key to continue...")

# plt.scatter(X_train, t_train, s=15)
# plt.scatter(X_valid, t_valid, s=15)
# plt.title("Training and Validation Set")
# plt.show()

# print(type(X_train_matrix))
# print(X_train_matrix)
# print(type(X_train))

# predictor_values = []
# for val in range(10):
#     m_val = 0
#     for i in range(len(X_train)):
#         predictor += w_const * (X_train[i] ** m_val)
#     predictor_values.append(predictor)
