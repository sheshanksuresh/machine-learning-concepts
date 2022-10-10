from re import M
import numpy as np
import matplotlib.pyplot as plt


def calculate_least_squares(feature_matrix, target_matrix, m_value):
    if m_value == 0:
        feature_matrix_transpose = np.transpose(feature_matrix)
        feature_multiply = np.matmul(feature_matrix_transpose, feature_matrix)
        featureT_multiply_target = np.matmul(feature_matrix_transpose, target_matrix)
        w_values = featureT_multiply_target / feature_multiply
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


def calculate_error(feature_matrix, target_matrix, w_values, m_value):
    if m_value == 0:
        prediction = feature_matrix * w_values
        pred_minus_target = np.subtract(prediction, target_matrix)
        number_of_features = feature_matrix.size
        prediction_error = (1 / number_of_features) * (
            np.matmul((np.transpose(pred_minus_target)), pred_minus_target)
        )
    else:
        prediction = np.matmul(feature_matrix, w_values)
        pred_minus_target = np.subtract(prediction, target_matrix)
        number_of_features = feature_matrix[:, 1:].size
        prediction_error = (1 / number_of_features) * (
            np.matmul((np.transpose(pred_minus_target)), pred_minus_target)
        )
    return prediction_error


def find_predictor_values(data_matrix, w_values, m_value):
    predictor = []
    predictor_val = 0
    for idx in range(len(data_matrix)):
        while m_value >= 0:
            predictor_val += w_values[idx] * X_train[idx] ** m_value
            m_value -= 1
        predictor.append(predictor_val)
    return predictor


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

# for m_val in range(10):
m_val = 2
print("*******************************")
print("*    STATISTICS FOR M = ", m_val, "   *")
print("*******************************\n")

feature_train_matrix = insert_m_features(feature_matrix=X_train_mat, m_value=m_val)
print("Feature Matrix for training set: \n", feature_train_matrix)
feature_valid_matrix = insert_m_features(feature_matrix=X_valid_mat, m_value=m_val)
print("Feature Matrix for validation set: \n", feature_valid_matrix)
w_train_values = calculate_least_squares(
    feature_matrix=feature_train_matrix, target_matrix=t_train_mat, m_value=m_val
)
print("W Value(s) for training set: \n", w_train_values)
w_valid_values = calculate_least_squares(
    feature_matrix=feature_valid_matrix, target_matrix=t_valid_mat, m_value=m_val
)
print("W Value(s) for validation set: \n", w_train_values)
training_error = calculate_error(
    feature_matrix=feature_train_matrix,
    target_matrix=t_train_mat,
    w_values=w_train_values,
    m_value=m_val,
)
print(f"The training error for M = {m_val} is: {training_error}")
validation_error = calculate_error(
    feature_matrix=feature_valid_matrix,
    target_matrix=t_valid_mat,
    w_values=w_valid_values,
    m_value=m_val,
)
print(f"The validation error for M = {m_val} is: {validation_error}")
training_predictor = find_predictor_values(
    data_matrix=X_train_mat, w_values=w_train_values, m_value=m_val
)
print("The predictor function values for the training set are: \n", training_predictor)
validation_predictor = find_predictor_values(
    data_matrix=X_valid_mat, w_values=w_valid_values, m_value=m_val
)
# print("The predictor function values for the validation set are: \n", validation_predictor)
plt.scatter(X_train, t_train, s=15)
plt.scatter(X_valid, t_valid, s=15)
m, b = np.polyfit(X_train, training_predictor, m_val)
plt.plot(X_train, training_predictor)
plt.plot(X_train, m*X_train+b)
plt.title(f"Plot for M = {m_val} Set")
plt.show()

next_m_val = m_val + 1
if m_val < 9:
    input(f"Press any key to continue to M = {next_m_val}")
else:
    print("Predictions and plots have been created.")
