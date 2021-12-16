
import autograd.numpy as np
from autograd import grad

import rebound
from utils import get_trajectories
import tqdm
import matplotlib.pyplot as plt
import plot as splt
from plot import animate_outcome2d, plot_trajectory, init_axes, plot_trajectories
from utils import *
from nbody import integrate, G
import matplotlib.pyplot as plt

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


def lambert(mu, r1_vec, r2_vec, delta_t, tol=1e-6):

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

primary = dict(
    m=1.0/G,
    p=(0.0, 0.0, 0.0),
    v=(0.0, 0.0, 0.0),
)

mu = 1.0
r1 = 1.0
r2 = 2.0

smaller_orbit = dict(
    m=0.0,
    p=(r1, 0.0, 0.0),
    v=(0.0, np.sqrt(mu/ r1), 0.0),
)

larger_orbit = dict(
    m=0.0,
    p=(0.0, r2, 0.0),
    v=(-1 * np.sqrt(mu/r2), 0.0, 0.0),
)

planets = [
    primary,
    smaller_orbit,
    larger_orbit,
]


T = 2 * np.pi * np.sqrt(r2**3 / mu)

errs = []
fds = []
thetas = []
Ts = np.linspace(T/10, T, 1000)
for t_max in tqdm.tqdm(Ts):
    _, p = integrate(planets, t_max, 100)
    target = p[2][-1]
    
    v, _ = lambert(
            mu, 
            np.array([r1, 0.0, 0.0]), 
            target,
            t_max, 
            tol=1e-8
    )
    
    planets[1]['v'] = (v[0], v[1], v[2])
    outcome, p1 = integrate(planets, t_max, 100)
    
    flight_distance = np.linalg.norm(p1[1][0] - target)
    err = np.linalg.norm(p1[1][-1] - target)
    
    fds.append(flight_distance)
    errs.append(err)

    mag = np.linalg.norm(v)
    theta = np.rad2deg(np.arccos(v[0] / mag))
    thetas.append(theta)

    #if err > 0.0005:
    #    plot_trajectories(p1, ["yellow", "green", "red"], [100, 30, 30])
    #    plt.savefig("outlier.png")

plt.clf()
plt.cla()
plt.yscale("log")
plt.scatter(Ts, errs)
plt.ylabel("Error")
plt.xlabel("Time of Flight")
plt.savefig("lambert_tof.png")

plt.clf()
plt.yscale("log")
plt.scatter(fds, errs)
plt.ylabel("Error")
plt.xlabel("Flight Distance")
plt.savefig("lambert_distance.png")

plt.clf()
plt.yscale("log")
plt.scatter(thetas, errs)
plt.ylabel("Error")
plt.xlabel("Launch Angle (Degrees)")
plt.savefig("lambert_angle.png")
