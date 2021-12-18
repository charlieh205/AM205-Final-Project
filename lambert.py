import autograd.numpy as np
from autograd import grad, hessian

# Implementation guided by Orbital Mechanics for Engnineering Students

# Stumpff Functions
def C(z):
    if z == 0:
        return 1 / 2
    elif z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    else:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)


def S(z):
    if z == 0:
        return 1 / 6
    if z > 0:
        return (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)) ** 3
    else:
        return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z)) ** 3


def newtons_method(z, F, tol, max_steps):

    dFdz = grad(F, argnum=0)

    update = 1e8
    i = 0
    for i in range(max_steps):
        update = F(z) / dFdz(z)
        z = z - update
        if np.abs(F(z)) < tol:
            return z, (i + 1)

    print("MAX STEPS REACHED")
    return z, (i + 1)


def secant_method(z, F, tol, max_steps, h=0.001):
    def dFdz(z):
        return (F(z) + F(z + h)) / (h)

    i = 0
    for i in range(max_steps):

        z -= F(z) / dFdz(z)

        if np.abs(F(z)) < tol:
            return z, (i + 1)

    print("MAX STEPS REACHED")
    return z, (i + 1)


def bisection_method(z, F, tol, max_steps, h=0.001):
    z_max = z

    z_min = z
    while F(z_min) > 0:
        z_min -= 0.1

    i = 0
    for i in range(max_steps):

        z = (z_max + z_min) / 2

        if np.abs(F(z)) < tol:
            return z, (i + 1)

        if F(z) > 0:
            z_max = z
        else:
            z_min = z

    print("MAX STEPS REACHED")
    return z, (i + 1)


def gradient_descent(z, F, tol, max_steps):
    dFdz = grad(lambda z: F(z) ** 2)

    i = 0
    for i in range(max_steps):
        z -= 0.1 * dFdz(z)
        if np.abs(F(z)) < tol:
            return z, (i + 1)

    print("MAX STEPS REACHED")
    return z, (i + 1)


def BFGS(z, F, tol, max_steps):
    dFdz = grad(lambda z: F(z) ** 2)

    i = 0
    for i in range(max_steps):
        z -= 0.1 * dFdz(z)
        if np.abs(F(z)) < tol:
            return z, (i + 1)

    print("MAX STEPS REACHED")
    return z, (i + 1)


def _lambert(
    root_finder, mu, r1_vec, r2_vec, delta_t, tol=1e-6, verbose=False, max_steps=5000
):

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    delta_theta = np.arccos(np.dot(r1_vec, r2_vec) / (r1 * r2))

    if np.cross(r1_vec, r2_vec)[-1] <= 0:
        delta_theta = 2 * np.pi - delta_theta

    A = np.sin(delta_theta) * np.sqrt((r1 * r2) / (1 - np.cos(delta_theta)))

    def y(z):
        return r1 + r2 + A * ((z * S(z) - 1) / np.sqrt(C(z)))

    def F(z):
        return ((y(z) / C(z)) ** 1.5) * S(z) + A * np.sqrt(y(z)) - np.sqrt(mu) * delta_t

    z = -100
    while F(z) < 0 or z <= 0:
        z += 0.1

    z, n_steps = root_finder(z, F, tol, max_steps)

    f = 1 - y(z) / r1
    g = A * np.sqrt(y(z) / mu)
    g_dot = 1 - y(z) / r2

    v_launch = 1 / g * (r2_vec - f * r1_vec)
    v_land = 1 / g * (g_dot * r2_vec - r1_vec)

    if verbose:
        return v_launch, v_land, n_steps

    return v_launch, v_land


def lambert(*args, **kwargs):
    return _lambert(newtons_method, *args, **kwargs)


def secant_lambert(*args, **kwargs):
    return _lambert(secant_method, *args, **kwargs)


def bisection_lambert(*args, **kwargs):
    return _lambert(bisection_method, *args, **kwargs)


def gradient_descent_lambert(*args, **kwargs):
    return _lambert(gradient_descent, *args, **kwargs)
