import autograd.numpy as np

import tqdm
import matplotlib.pyplot as plt

from nbody import integrate, G
from lambert import *

primary = dict(
    m=1.0 / G,
    p=(0.0, 0.0, 0.0),
    v=(0.0, 0.0, 0.0),
)

mu = 1.0
r1 = 1.0
r2 = 2.0

smaller_orbit = dict(
    m=0.0,
    p=(r1, 0.0, 0.0),
    v=(0.0, np.sqrt(mu / r1), 0.0),
)

larger_orbit = dict(
    m=0.0,
    p=(0.0, r2, 0.0),
    v=(-1 * np.sqrt(mu / r2), 0.0, 0.0),
)

planets = [
    primary,
    smaller_orbit,
    larger_orbit,
]

T = 2 * np.pi * np.sqrt(r2 ** 3 / mu)

newton = []
secant = []
bisection = []
gradient_descent = []

Ts = np.linspace(T / 10, T, 100)
for t_max in tqdm.tqdm(Ts):
    _, p = integrate(planets, t_max, 100)
    target = p[2][-1]

    _, _, n_steps = lambert(
        mu,
        np.array([r1, 0.0, 0.0]),
        target,
        t_max,
        tol=1e-12,
        verbose=True,
    )
    newton.append(n_steps)

    _, _, n_steps = secant_lambert(
        mu,
        np.array([r1, 0.0, 0.0]),
        target,
        t_max,
        tol=1e-12,
        verbose=True,
    )
    secant.append(n_steps)

    _, _, n_steps = bisection_lambert(
        mu,
        np.array([r1, 0.0, 0.0]),
        target,
        t_max,
        tol=1e-12,
        verbose=True,
    )
    bisection.append(n_steps)

    _, _, n_steps = gradient_descent_lambert(
        mu,
        np.array([r1, 0.0, 0.0]),
        target,
        t_max,
        tol=1e-12,
        verbose=True,
    )
    gradient_descent.append(n_steps)

names = ["Newton's Method", "Secant Method", "Bisection Method", "Gradient Descent"]
results = [newton, secant, bisection, gradient_descent]
for (n, r) in zip(names, results):
    min_ = round(np.min(r), 2)
    max_ = round(np.max(r), 2)
    mean_ = round(np.mean(r), 2)
    var = round(np.var(r), 2)
    print(f"{n} & {min_} & {mean_} & {max_} & {var} \\\\")
