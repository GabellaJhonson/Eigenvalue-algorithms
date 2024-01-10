import numpy as np

epsilon = 10 ** (-5)


def sum(matrix):
    mask = np.eye(matrix.shape[0], dtype=bool)
    masked_matrix = np.ma.masked_array(matrix, mask)

    sum_of_squares = np.sum(np.square(masked_matrix))
    if sum_of_squares < 0:
        sum_of_squares = - sum_of_squares
    print(f"{sum_of_squares} <= {epsilon}")
    return sum_of_squares

def max(matrix):

    mask = np.eye(matrix.shape[0], dtype=bool)
    masked_matrix = np.ma.masked_array(matrix, mask)

    absolute_values = np.abs(masked_matrix)

    max_index = np.argmax(absolute_values)

    return max_index

def rotation_method(matrix):
    len = matrix.shape[0]
    counter = 0
    A = matrix.dot(matrix.transpose())
    #A = matrix

    eigenvectors = np.eye(len)

    while (sum(A) > epsilon):
        print(f"\n--------------iteration {counter}------------------\n")
        print(f"\n{A}\n")
        max_index_row, max_index_col = np.unravel_index(max(A), A.shape)

        print("Максимальный элемент:", A[max_index_row, max_index_col])
        print("Индексы максимального элемента:", max_index_row, max_index_col)

        tg2 = (2 * A[max_index_row][max_index_col]) / (A[max_index_row][max_index_row] - A[max_index_col][max_index_col])
        print(f"tg:{tg2}")
        cos2 = (1 + tg2 ** 2) ** (-0.5)
        cos = ((1 + cos2) / 2) ** 0.5
        sin = ((1 - cos2) / 2) ** 0.5
        if tg2 < 0:
            sin = -sin

        T = np.eye(len)
        T[max_index_row][max_index_row] = cos
        T[max_index_col][max_index_col] = cos
        T[max_index_col][max_index_row] = sin
        T[max_index_row][max_index_col] = -sin

        T_transpose = T.transpose()
        print(f"\n{T}\n")
        A = T_transpose.dot(A)
        A = A.dot(T)
        print(f"\n{A}\n")
        counter +=1

        A[max_index_row][max_index_col] = 0
        A[max_index_col][max_index_row] = 0
        eigenvectors = eigenvectors.dot(T)

    print(f"\n-------------- Result ------------------\n")
    matrix = matrix.dot(matrix.transpose())

    for i in range(matrix.shape[0]):
        test_matrix = np.eye(len).dot(A[i][i])
        result = matrix.dot(eigenvectors[:, i]) - test_matrix.dot(eigenvectors[:, i])
        print(f"Собственное значение:{A[i][i]}, {eigenvectors[:, i].transpose()}")
        print(f"Невязка r :{np.linalg.norm(np.dot(matrix, eigenvectors[:, i]) - A[i][i]*eigenvectors[:, i], 1)}")
    # print("Вектора", eigenvectors)
    return A
A = np.array([[0.6444, 0.0000, -0.1683, 0.1184, 0.1973],
              [-0.0395, 0.4208, 0.0000, -0.0802, 0.0263],
              [0.0132, -0.1184, 0.7627, 0.0145, 0.0460],
              [0.0395, 0.0000, -0.0960, 0.7627, 0.0000],
              [0.0263, -0.0395, 0.1907, -0.0158, 0.5523]])
print(rotation_method(A))
# 0.7808349384723945