import numpy as np
import scipy as sc
from scipy.optimize import minimize_scalar, fmin

def matrix_multiplication(matrix1, matrix2):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    def create_matrix(n: int, p: int) -> list:
        matrix = [[0] * p for _ in range(n)]
        return matrix

    if len(matrix1[0]) != len(matrix2):
        # print("impossible to make an multiplication")
        raise ValueError(
            "impossible to make an multiplication")
    else:
        # print("possible to multiply! yeeei!!!")
        m = len(matrix1)
        n = len(matrix2)
        p = len(matrix2[0])
        mult_matrix = create_matrix(m, p)

        for i in range(m):
            for k in range(p):
                curr_sum = 0
                for j in range(n):
                    curr_sum += matrix1[i][j] * matrix2[j][k]
                mult_matrix[i][k] = curr_sum
        return mult_matrix


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a1 = list(map(float, a_1.split()))
    a2 = list(map(float, a_2.split()))

    if a1 == a2:  # inf solutions
        return None

    def f(x):
        return a1[0] * (x ** 2) + a1[1] * x + a1[2]

    def p(x):
        return a2[0] * (x ** 2) + a2[1] * x + a2[2]
    
    roots = []

    for x in range(-100, 101):
        f_value = f(x)
        p_value = p(x)
        if abs(f_value - p_value) < 0.01:
            roots.append((x, round(f_value, 2)))

    return roots


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    m3 = np.sum((x - mean) ** 3) / n
    skewness = m3 / std ** 3
    return round(skewness, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    m4 = np.sum((x - mean) ** 4) / n
    excess = m4 / std ** 4 - 3
    return round(excess, 2)
