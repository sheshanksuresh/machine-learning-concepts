import numpy as np
import matplotlib.pyplot as plt


def calculate_least_squares(feature_matrix, target_matrix, m_value):
    """Calculate lease squares regression for provided feature and target matrices.

    Args:
        feature_matrix (ndarray): _description_
        target_matrix (_type_): _description_
        m_value (_type_): _description_

    Returns:
        _type_: _description_
    """
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


def find_predictor_values(X_matrix, w_values, m_value):
    if m_value == 0:
        prediction = X_matrix * w_values
    else:
        prediction = np.matmul(X_matrix, w_values)
    return prediction


def plot(
    m_value,
    X_train,
    X_valid,
    training_set,
    validation_set,
    training_prediction,
    validation_prediction,
    f_true_training,
    f_true_validation,
):
    plt.scatter(X_train, training_set, s=15, c="b", label="Training Set")
    plt.scatter(X_valid, validation_set, s=15, c="orange", label="Validation Set")
    plt.plot(
        X_valid, validation_prediction, c="g", label="Validation Prediction (f_M_(x))"
    )
    plt.plot(
        X_valid, f_true_validation, c="r", label="True Curve Validation (f_true_(x))"
    )
    plt.legend()
    plt.title(f"Plot for M = {m_value}")
    plt.show()
    return


def main():
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

    max_M_val = 10
    for m_val in range(max_M_val):
        print("*******************************")
        print("*    STATISTICS FOR M = ", m_val, "   *")
        print("*******************************\n")

        feature_train_matrix = insert_m_features(
            feature_matrix=X_train_mat, m_value=m_val
        )
        print("Feature Matrix for training set: \n", feature_train_matrix)
        feature_valid_matrix = insert_m_features(
            feature_matrix=X_valid_mat, m_value=m_val
        )
        print("Feature Matrix for validation set: \n", feature_valid_matrix)
        w_train_values = calculate_least_squares(
            feature_matrix=feature_train_matrix,
            target_matrix=t_train_mat,
            m_value=m_val,
        )
        print("W Value(s) for training set: \n", w_train_values)
        w_valid_values = calculate_least_squares(
            feature_matrix=feature_valid_matrix,
            target_matrix=t_valid_mat,
            m_value=m_val,
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
        training_prediction = find_predictor_values(
            X_matrix=feature_train_matrix, w_values=w_train_values, m_value=m_val
        )
        print(
            "The predictor function values for the training set are: \n",
            training_prediction,
        )
        validation_prediction = find_predictor_values(
            X_matrix=feature_valid_matrix, w_values=w_valid_values, m_value=m_val
        )
        print(
            "The predictor function values for the validation set are: \n",
            validation_prediction,
        )
        f_true_train = np.sin(4 * np.pi * X_train)
        f_true_valid = np.sin(4 * np.pi * X_valid)

        next_m_val = m_val + 1
        plot(
            m_value=m_val,
            X_train=X_train,
            X_valid=X_valid,
            training_set=t_train,
            validation_set=t_valid,
            training_prediction=training_prediction,
            validation_prediction=validation_prediction,
            f_true_training=f_true_train,
            f_true_validation=f_true_valid,
        )


if __name__ == "__main__":
    main()
