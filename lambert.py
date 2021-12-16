
import autograd.numpy as np
from autograd import grad

# Implementation guided by Orbital Mechanics for Engnineering Students

# Stumpff Functions
def C(z):
    if z == 0:
        return 1/2
    elif z > 0:
        return (1 - np.cos(np.sqrt(z)))/z
    else:
        return (np.cosh(np.sqrt(-z)) - 1)/(-z)

def S(z):
    if z == 0:
        return 1/6
    if z > 0:
        return (np.sqrt(z) - np.sin(np.sqrt(z)))/(np.sqrt(z))**3
    else:
        return (np.sinh(np.sqrt(-z)) - np.sqrt(-z))/(np.sqrt(-z))**3


def lambert(mu, r1_vec, r2_vec, delta_t, tol=1e-6, verbose=False):

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    delta_theta = np.arccos(np.dot(r1_vec, r2_vec) / (r1 * r2))

    if np.cross(r1_vec, r2_vec)[-1] <= 0:
        delta_theta = 2 * np.pi - delta_theta

    A = np.sin(delta_theta) * np.sqrt((r1 * r2) / (1 - np.cos(delta_theta)))

    def y(z):
        return r1 + r2 + A * ((z * S(z) - 1) / np.sqrt(C(z)))

    def F(z, t):
        return ((y(z) / C(z)) ** 1.5) * S(z) + A * np.sqrt(y(z)) - np.sqrt(mu) * t

    dFdz = grad(F, argnum=0)

    z = -100
    while F(z, delta_t) < 0 or z <= 0:
        z += 0.1
    
    nmax = 5000

    ratio = 1
    n = 0
    while (np.abs(ratio) > tol) and (n <= nmax):
        n += 1
        ratio = F(z, delta_t) / dFdz(z, delta_t)
        z = z - ratio
    
    if n >= nmax:
        print("Max iterations hit")

    f = 1 - y(z) / r1
    g = A * np.sqrt(y(z)/mu)
    g_dot = 1 - y(z) / r2
    
    v_launch = 1/g * (r2_vec - f * r1_vec)
    v_land = 1/g * (g_dot*r2_vec - r1_vec)
    return v_launch, v_land
