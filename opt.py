#!/usr/bin/env python3

from collections import namedtuple


Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов модельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива — менее nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""


df = pd.read.csv('jla_mub.txt', sep=' ')

z = pd.z

c = 3 * 10 ** 11



def f_i(z_i, omega):
    return 1 / np.sqrt( (1 - omega)*(1 + z_i)**3 + omega)

def f_i_omega(z_i, omega):
    (1-(z_i + 1)**3 )* (1 / (- 2 * (np.sqrt( (1 - omega)*(1 + z_i)**3 + omega))**(3/2)))
    
def f(Z,  H0, omega):
    mass = []
    for z in Z:
        mass.append( c *integrate.quad(f_i, 0, z, (omega))[0] * ((1 + z)  / H0))
    return np.array(mass)

def f_omega(Z,  H0, omega):
    mass = []
    for z in Z:
        mass.append( c *integrate.quad(f_i_omega, 0, z, (omega))[0] * ((1 + z)  / H0))
    return np.array(mass)

def f_H0(Z,  H0, omega):
    mass = []
    for z in Z:
        mass.append( - c *integrate.quad(f_i_omega, 0, z, (omega))[0] * ((1 + z)  / H0**2))
    return np.array(mass)

def dmu(mu):
    return 10**(0.2 * mu + 1)

d=dmu(df.mu)

def j(z, h0, omega):
    jac = np.empty((z.values.size, 2))
    jac[: , 0] = f_omega(z, h0, omega)
    jac[: , 1] = f_H0(z, h0, omega)
    return jac

def gauss_newton(y, f, j, x, k=1, tol=1e-4, max iter = 1000):
    i = 0
    cost = []
    while True:
        i+=1
        r = y - f(*x)
        cost.append(0.5 * np.dot(r, r))
        jac = j(*x)
        g = np.dot(jac.T, r)
        g_norm = np.linalg.norm(g)
        delta_x = np.linalg.solve(jac.T @ jac, - g)
        x = x + k * delta_x
        if i >= max_iter:
            break
        if np.linalg.norm(delta_x) <= tol * np.linalg.norm(x):
            break
    cost = np.array(cost)
    return Result(nfev=i, cost=cost, grandnorm=g_norm, x=x)

r = gauss_newton(d, lambda *args: f(z, *args), lambda *args: j(z, *args), (50, 0.5), k=0.5, tol=1e-3, max_iter=1000)
r

plt.figure(dpi=80)
x = np.linspace(0, 1, 100)
y = f(x, *r.x)
plt.plot(df.z, df.mu, '.', label='data', color='red')
plt.plot(x, 5 * np.log10(y) - 5, label='fit', color='yellow')
plt.show()

plt.figure(dpi=80)
plt.plot(r.cost)
plt.show()

def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4):
    pass


if __name__ == "__main__":
    pass
