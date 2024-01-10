import numpy as np

epsilon = 10 ** (-5)


def power_method(matrix):
    counter = 0
    A = matrix.dot(matrix.transpose())
    # A = matrix

    A_power = A
    y_0 = np.array([1, 0, 0, 0, 0]).transpose()
    z_k = y_0

    lambda_1 = np.dot(A.dot(z_k), z_k) / np.dot(z_k, z_k)
    lambda_ = lambda_1 - 2 * epsilon

    while abs(lambda_1 - lambda_) > epsilon:
        print(f"{abs(lambda_1 - lambda_)} <= {epsilon}")
        print(f"\n------------------iteration {counter}------------------\n")
        print("z:")
        A_power = A_power.dot(A)
        norm = np.linalg.norm(A_power.dot(y_0), ord=1)
        z_k = A_power.dot(y_0)
        print(z_k)
        print(f"norm:{norm}")
        z_k = z_k.dot((norm) ** (-1))
        print(z_k)
        lambda_ = lambda_1
        print("lambda:")
        print(lambda_)
        lambda_1 = np.dot(A.dot(z_k), z_k) / np.dot(z_k, z_k)
        print(f"{lambda_1}\n")
        counter += 1

    print(f"{abs(lambda_1 - lambda_)} <= {epsilon}")
    print(f"\n-------------- Result ------------------\n")
    matrix = matrix.dot(matrix.transpose())
    test_matrix = np.eye(5).dot(lambda_)
    print(matrix.dot(z_k).transpose())
    print(test_matrix.dot(z_k).transpose())
    matrix = matrix.dot(z_k)
    test_matrix = test_matrix.dot(z_k)
    result = matrix - test_matrix

    print(f"Собственное значение:{lambda_}, {z_k.transpose()}")
    print(f"Невязка r :{result.transpose()}")
    print(np.linalg.norm(result.transpose(), 1))

    return A
A = np.array([[0.6444, 0.0000, -0.1683, 0.1184, 0.1973],
              [-0.0395, 0.4208, 0.0000, -0.0802, 0.0263],
              [0.0132, -0.1184, 0.7627, 0.0145, 0.0460],
              [0.0395, 0.0000, -0.0960, 0.7627, 0.0000],
              [0.0263, -0.0395, 0.1907, -0.0158, 0.5523]])
power_method(A)