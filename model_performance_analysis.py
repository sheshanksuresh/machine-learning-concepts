import numpy as np
import matplotlib.pyplot as plt


def calculate_least_squares(feature_matrix, target_matrix, m_value):
    if m_value == 0:
        feature_matrix_transpose = np.transpose(feature_matrix)
        feature_multiply = np.matmul(feature_matrix_transpose, feature_matrix)
        featureT_multiply_target = np.matmul(feature_matrix_transpose, target_matrix)
        w_values = np.matmul(np.matrix(1 / feature_multiply), featureT_multiply_target)
    else:
        feature_matrix_transpose = np.transpose(feature_matrix)
        feature_multiply = np.matmul(feature_matrix_transpose, feature_matrix)
        featureT_multiply_target = np.matmul(feature_matrix_transpose, target_matrix)
        w_values = np.matmul(np.linalg.inv(feature_multiply), featureT_multiply_target)
    return w_values


def insert_m_features(feature_matrix, m_value):
    if m_value == 0:
        return feature_matrix[:, 0]
    elif m_value == 1:
        return feature_matrix
    else:
        for i in range(2, m_value + 1):
            max_column_idx = feature_matrix.shape[1]
            feature_matrix = np.insert(
                feature_matrix, max_column_idx, feature_matrix[:, 1] ** i, axis=1
            )
        return feature_matrix


def calculate_error(feature_matrix, target_matrix, w_values):
    prediction = np.matmul(feature_matrix, w_values)
    pred_minus_target = np.subtract(prediction, target_matrix)
    number_of_features = feature_matrix[:, 1:].size
    prediction_error = (1 / number_of_features) * (
        np.matmul((np.transpose(pred_minus_target)), pred_minus_target)
    )
    return prediction_error


# def main():

# Code for Data Generation
X_train = np.linspace(0.0, 1.0, 10)  # training set
X_valid = np.linspace(0.0, 1.0, 100)  # validation set
np.random.seed(2679)
t_valid = np.sin(4 * np.pi * X_valid) + 0.3 * np.random.randn(100)
t_train = np.sin(4 * np.pi * X_train) + 0.3 * np.random.randn(10)

# Append a 1's column vector to training and validation matrices and convert to a 2D matrix
X_train_mat = np.reshape(X_train, (-1, 1))
X_train_mat = np.insert(X_train_mat, 0, np.ones(10), axis=1)
X_valid_mat = np.reshape(X_valid, (-1, 1))
X_valid_mat = np.insert(X_valid_mat, 0, np.ones(100), axis=1)

# Transform target matrices into 2D matrices
t_train_mat = np.reshape(t_train, (-1, 1))
t_valid_mat = np.reshape(t_valid, (-1, 1))

for m_val in range(10):
    print("******************************************")
    print("*    STARTING CALCULATIONS FOR M = ", m_val, "   *")
    print("******************************************")
    
    feature_train_matrix = insert_m_features(X_train_mat, m_value=m_val)
    feature_valid_matrix = insert_m_features(X_valid_mat, m_value=m_val)
    w_train_values = calculate_least_squares(feature_train_matrix, t_train_mat, m_value=m_val)
    w_valid_values = calculate_least_squares(feature_valid_matrix, t_valid_mat, m_value=m_val)
    training_error = calculate_error(feature_train_matrix, t_train_mat, w_train_values)
    validation_error = calculate_error(feature_valid_matrix, t_train_mat, w_valid_values)

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
