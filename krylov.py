import numpy as np
from sympy import Symbol, solve

def krylov(A, n):
    a = np.array(A)
    At = a.transpose()
    print("Симметрическая A*A^T:\n\t", At)
    a = np.dot(At, a)

    c = []
    c.append([1, 0, 0, 0, 0]) 
    for i in range(1, n + 1):
        c.append(np.dot(a, c[i - 1])) 

    print("\nВекторы C^i:")
    for el in c:
        print("\t", el)

    C = np.array(c) 
    cn = c.pop() 
    c = np.array(c).transpose()
    for i in range(n):
        c[i] = list(reversed(c[i]))
    p = np.linalg.solve(c, cn)

    print("\nКоэффициенты собственного многочлена P(lambda):\n\t", p) 

    x = Symbol('x')
    Lambda = solve(x**5 - p[0] * x**4 - p[1] * x**3 - p[2] * x**2 - p[3] * x - p[4], x)
    print("\nСобственные значения lambda:\n\t", Lambda)

    l = max(Lambda)
    print("\nМаксимальное собственное lambdaMax:\n\t", l)

    b = np.ones(n)
    for i in range(1, n):
        b[i] = b[i - 1] * l - p[i - 1]
    print("\nКоэффиценты B^i:\n\t", b)

    x = np.sum([b[i] * C[n - i - 1] for i in range(n)], axis=0)
    print("\nСобственный вектор матрицы A - x(lambdaMax):\n\t", x)

    r = np.dot(a, x) - l * x 
    print("\nВектор невязки r:\n\t", r)

    rnorm = np.linalg.norm(r, 1)
    print("\nНорма невязки ||r||:\n\t", rnorm) # считаем невязку собственного многочлена �%(�F)
    p = np.insert(p, 0, -1)
    r1 = sum(-(l ** (n - i)) * p[i] for i in range(n + 1))
    print("\nНорма невязки многочлена ", r1)
A = np.array([[0.6444, 0.0000, -0.1683, 0.1184, 0.1973],
              [-0.0395, 0.4208, 0.0000, -0.0802, 0.0263],
              [0.0132, -0.1184, 0.7627, 0.0145, 0.0460],
              [0.0395, 0.0000, -0.0960, 0.7627, 0.0000],
              [0.0263, -0.0395, 0.1907, -0.0158, 0.5523]])
n = 5
krylov(A,n)