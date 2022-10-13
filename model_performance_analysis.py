import numpy as np
import matplotlib.pyplot as plt


def calculate_w_value(feature_matrix, target_matrix, m_value):
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


def calculate_B_matrix(lambda_val):
    B_matrix = np.zeros((10, 10))
    for i in range(1, 10):
        B_matrix[i][i] = 2 * lambda_val
    return B_matrix


def calculate_w9_regularization(feature_matrix, target_matrix, B_matrix):
    N = len(target_matrix)
    w_first_half = (
        np.matmul(np.transpose(feature_matrix), feature_matrix) + N * B_matrix / 2
    )
    if np.linalg.det(w_first_half) != 0:
        w_first_half = np.linalg.inv(w_first_half)
    else:
        print("Determinant is equal to zero")
        exit()
    w_second_half = np.matmul(np.transpose(feature_matrix), target_matrix)
    w_value_M9 = np.matmul(w_first_half, w_second_half)
    return w_value_M9


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
        N_value = len(target_matrix)
        prediction_error = (1 / N_value) * (
            np.matmul((np.transpose(pred_minus_target)), pred_minus_target)
        )
    else:
        prediction = np.matmul(feature_matrix, w_values)
        pred_minus_target = np.subtract(prediction, target_matrix)
        N_value = len(target_matrix)
        prediction_error = (1 / N_value) * (
            np.matmul((np.transpose(pred_minus_target)), pred_minus_target)
        )
    return prediction_error[0][0]


def find_predictor_values(X_matrix, w_values, m_value):
    if m_value == 0:
        prediction = X_matrix * w_values
    else:
        prediction = np.matmul(X_matrix, w_values)
    return prediction


def plot_M(
    m_value,
    X_train,
    X_valid,
    training_set,
    validation_set,
    validation_prediction,
    f_true_validation,
):
    plt.figure(figsize=(10, 8))
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


def plot_M_regularization(
    m_value,
    X_train,
    X_valid,
    training_set,
    validation_set,
    validation_prediction,
    f_true_validation,
    lambda_val
):
    plt.figure(figsize=(10, 8))
    plt.scatter(X_train, training_set, s=15, c="b", label="Training Set")
    plt.scatter(X_valid, validation_set, s=15, c="orange", label="Validation Set")
    plt.plot(
        X_valid,
        validation_prediction,
        c="g",
        label="Validation Prediction Regularization (f_M_(x))",
    )
    plt.plot(
        X_valid, f_true_validation, c="r", label="True Curve Validation (f_true_(x))"
    )
    plt.legend()
    plt.title(f"Plot for M = {m_value} with Regularization and Lambda = {lambda_val}")
    plt.show()
    return


def plot_errors(
    training_errors,
    validation_errors,
    lambda_training_error_1,
    lambda_training_error_2,
    lambda_validation_error_1,
    lambda_validation_error_2,
    m_values,
):
    plt.figure(figsize=(10, 8))
    plt.scatter(m_values, training_errors, s=10, c="b", label="Training Error")
    plt.scatter(m_values, validation_errors, s=10, c="orange", label="Validation Error")
    plt.scatter(9, lambda_training_error_1, s=10, c="r", label="Lambda 1 Training Error")
    plt.scatter(9, lambda_training_error_2, s=10, c="g", label="Lambda 2 Training Error")
    plt.scatter(9, lambda_validation_error_1, s=10, c="m", label="Lambda 1 Validation Error")
    plt.scatter(9, lambda_validation_error_2, s=10, c="c", label="Lambda 2 Validation Error")
    plt.legend()
    plt.title("Training and Validation Errors")
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

    f_true_valid = np.sin(4 * np.pi * X_valid)

    max_M_val = 10
    m_values = np.linspace(0, 9, 10)
    training_errors = []
    validation_errors = []

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
        
        w_train_values = calculate_w_value(
            feature_matrix=feature_train_matrix,
            target_matrix=t_train_mat,
            m_value=m_val,
        )
        print("W Value(s) for training set: \n", w_train_values)

        training_error = calculate_error(
            feature_matrix=feature_train_matrix,
            target_matrix=t_train_mat,
            w_values=w_train_values,
            m_value=m_val,
        )
        training_errors.append(training_error)
        print(f"The training error for M = {m_val} is: {training_error}")

        validation_error = calculate_error(
            feature_matrix=feature_valid_matrix,
            target_matrix=t_valid_mat,
            w_values=w_train_values,
            m_value=m_val,
        )
        validation_errors.append(validation_error)
        print(f"The validation error for M = {m_val} is: {validation_error}")

        validation_prediction = find_predictor_values(
            X_matrix=feature_valid_matrix, w_values=w_train_values, m_value=m_val
        )
        print(
            "The predictor function values for the validation set are: \n",
            validation_prediction,
        )

        plot_M(
            m_value=m_val,
            X_train=X_train,
            X_valid=X_valid,
            training_set=t_train,
            validation_set=t_valid,
            validation_prediction=validation_prediction,
            f_true_validation=f_true_valid,
        )
        if m_val == 9:
            lambda_value_1 = 10**-10
            lambda_value_2 = 10**3

            B_lambda_1 = calculate_B_matrix(lambda_val=lambda_value_1)
            B_lambda_2 = calculate_B_matrix(lambda_val=lambda_value_2)

            w_train_values_l1 = calculate_w9_regularization(
                feature_matrix=feature_train_matrix,
                target_matrix=t_train_mat,
                B_matrix=B_lambda_1,
            )
            w_train_values_l2 = calculate_w9_regularization(
                feature_matrix=feature_train_matrix,
                target_matrix=t_train_mat,
                B_matrix=B_lambda_2,
            )

            training_error_l1 = calculate_error(
                feature_matrix=feature_train_matrix,
                target_matrix=t_train_mat,
                w_values=w_train_values_l1,
                m_value=m_val,
            )
            lambda_training_error_1 = training_error_l1
            training_error_l2 = calculate_error(
                feature_matrix=feature_train_matrix,
                target_matrix=t_train_mat,
                w_values=w_train_values_l2,
                m_value=m_val,
            )
            lambda_training_error_2 = training_error_l2
            validation_error_l1 = calculate_error(
                feature_matrix=feature_valid_matrix,
                target_matrix=t_valid_mat,
                w_values=w_train_values_l1,
                m_value=m_val,
            )
            lambda_validation_error_1 = validation_error_l1
            validation_error_l2 = calculate_error(
                feature_matrix=feature_valid_matrix,
                target_matrix=t_valid_mat,
                w_values=w_train_values_l2,
                m_value=m_val,
            )
            lambda_validation_error_2 = validation_error_l2

            validation_prediction_l1 = find_predictor_values(
                X_matrix=feature_valid_matrix, w_values=w_train_values_l1, m_value=m_val
            )
            print(
                "The predictor function values for the validation set are: \n",
                validation_prediction_l1,
            )
            validation_prediction_l2 = find_predictor_values(
                X_matrix=feature_valid_matrix, w_values=w_train_values_l2, m_value=m_val
            )
            print(
                "The predictor function values for the validation set are: \n",
                validation_prediction_l2,
            )

            plot_M_regularization(
                m_value=m_val,
                X_train=X_train,
                X_valid=X_valid,
                training_set=t_train,
                validation_set=t_valid,
                validation_prediction=validation_prediction_l1,
                f_true_validation=f_true_valid,
                lambda_val=lambda_value_1
            )
            plot_M_regularization(
                m_value=m_val,
                X_train=X_train,
                X_valid=X_valid,
                training_set=t_train,
                validation_set=t_valid,
                validation_prediction=validation_prediction_l2,
                f_true_validation=f_true_valid,
                lambda_val=lambda_value_2
            )

    plot_errors(
        training_errors=training_errors,
        validation_errors=validation_errors,
        lambda_training_error_1=lambda_training_error_1,
        lambda_training_error_2=lambda_training_error_2,
        lambda_validation_error_1=lambda_validation_error_1,
        lambda_validation_error_2=lambda_validation_error_2,
        m_values=m_values
    )


if __name__ == "__main__":
    main()
