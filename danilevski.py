import numpy as np
from sympy import symbols, solve

A = np.array([[0.6444, 0.0000, -0.1683, 0.1184, 0.1973],
              [-0.0395, 0.4208, 0.0000, -0.0802, 0.0263],
              [0.0132, -0.1184, 0.7627, 0.0145, 0.0460],
              [0.0395, 0.0000, -0.0960, 0.7627, 0.0000],
              [0.0263, -0.0395, 0.1907, -0.0158, 0.5523]])
print("Матрица A:\n\t", A)

At = A.transpose()

a = np.dot(At, A)
print("\nСимметрическая матрица A*A^T:\n\t", a)

n = len(A)
f = a
s = np.identity(n)

for i in range(n - 1):
    m = np.identity(n)
    m[n - 2 - i] = f[n - 1 - i]

    f = np.dot(m, f)
    f = np.dot(f, np.linalg.inv(m))
    s = np.dot(s, np.linalg.inv(m))

print("\nКаноническая Ф:\n\t", f)
print("\nМатрица преобразования S:\n\t", s)

p = f[0]
print("\nКоэффициенты собственного многочлена P(lambda):\n\t", p)

x = symbols('x')
Lambda = solve(x**5 - p[0] * x**4 - p[1] * x**3 - p[2] * x**2 - p[3] * x - p[4], x)
print("\nСобственные значения lambda:\n\t", Lambda)

maxLambda = max(Lambda)
print("\nmax(lambda):\n\t", maxLambda)
for i in Lambda:
    y = [i ** j for j in range(n - 1, -1, -1)]
    y = np.dot(s, y)
    print("\nСобственный вектор матрицы \n\t", y)
    r = np.dot(a, y) - maxLambda * y
    print("\nВектор невязки r:\n\t", r)
p = np.insert(p, 0, -1)
r1 = sum(-(maxLambda ** (n - i)) * p[i] for i in range(n + 1))
print("\nНорма невязки многочлена ", r1)
rnorm = np.linalg.norm([-2.52242671194836e-13,-1.43152156795168e-12,4.08562073062058e-13,
-2.32880381645373e-12, 1.88737914186277e-15], 1)
print("\nНорма невязки многочлена ", rnorm)
