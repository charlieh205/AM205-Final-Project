
from dataclasses import dataclass

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from plot import animate_outcome3d

# km, kg, s
sun = {
    "m": 1.9884754159566474e30,
    "p": (130.12022433639504, 493.88921553478576, 1.9894885155190423),
    "v": (-9.08252417469984e-05, 2.763470714263038e-05, 9.333023252795563e-08),
}
earth = {
    "m": 6.045825576341311e24,
    "p": (-52532397.16477036, -142290670.7988091, 6714.9380375893525),
    "v": (27.459556210605758, -10.428715463342813, 0.0004283693350210454),
}

mars = {
    "m": 6.417120534329241e23,
    "p": (91724696.20692892, -189839018.6923888, -6228099.232650615),
    "v": (22.733051422552098, 12.621328917003236, -0.29323856116219776),
}

init = [sun, earth, mars]
G = 6.67408e-20

def get_n_bodies(s):
    return s.shape[-1] // 7

def get_m(s, i):
    return s[i]

def get_x(s, i):
    num_bodies = get_n_bodies(s)
    return s[num_bodies + i]

def get_y(s, i):
    num_bodies = get_n_bodies(s)
    return s[2 * num_bodies + i]

def get_z(s, i):
    num_bodies = get_n_bodies(s)
    return s[3 * num_bodies + i]

def get_vx(s, i):
    num_bodies = get_n_bodies(s)
    return s[4 * num_bodies + i]

def get_vy(s, i):
    num_bodies = get_n_bodies(s)
    return s[5 * num_bodies + i]

def get_vz(s, i):
    num_bodies = get_n_bodies(s)
    return s[6 * num_bodies + i]

def get_position(s, i):

    x = get_x(s, i)
    y = get_y(s, i)
    z = get_z(s, i)

    return np.array([x, y, z])

def get_r(s, i, j):
    p1 = get_position(s, i)
    p2 = get_position(s, j)
    r = np.linalg.norm(p2 - p1)
    return r

def x_accel(s, i, j):
    x1 = get_x(s, i)
    x2 = get_x(s, j)
    m2 = get_m(s, j)

    r = get_r(s, i, j)
    return m2 * (x2 - x1) / (r) ** (3)


def y_accel(s, i, j):
    y1 = get_y(s, i)
    y2 = get_y(s, j)
    m2 = get_m(s, j)

    r = get_r(s, i, j)
    return m2 * (y2 - y1) / (r) ** (3)


def z_accel(s, i, j):
    z1 = get_z(s, i)
    z2 = get_z(s, j)
    m2 = get_m(s, j)

    r = get_r(s, i, j)
    return m2 * (z2 - z1) / (r) ** (3)


def xpp(s, i):
    num_bodies = get_n_bodies(s)
    other_indices = list(range(num_bodies))
    other_indices.remove(i)
    return G * sum([x_accel(s, i, j) for j in other_indices])


def ypp(s, i):
    num_bodies = get_n_bodies(s)
    other_indices = list(range(num_bodies))
    other_indices.remove(i)
    return G * sum([y_accel(s, i, j) for j in other_indices])


def zpp(s, i):
    num_bodies = get_n_bodies(s)
    other_indices = list(range(num_bodies))
    other_indices.remove(i)
    return G * sum([z_accel(s, i, j) for j in other_indices])


def F(s, t):
    num_bodies = get_n_bodies(s)
    return np.concatenate(
        (
            np.zeros_like(s[4 * num_bodies : 5 * num_bodies]),
            s[4 * num_bodies : 5 * num_bodies],
            s[5 * num_bodies : 6 * num_bodies],
            s[6 * num_bodies : 7 * num_bodies],
            [xpp(s, i) for i in range(num_bodies)],
            [ypp(s, i) for i in range(num_bodies)],
            [zpp(s, i) for i in range(num_bodies)],
        )
    )


def to_positions(solution):
    num_bodies = get_n_bodies(solution)
    outcome = [
        np.stack(
            [
                solution[:, num_bodies + i],
                solution[:, 2 * num_bodies + i],
                solution[:, 3 * num_bodies + i],
            ],
            axis=1,
        )
        for i in range(num_bodies)
    ]
    return np.stack(outcome)

def planet_dict_to_vector(init):
    m = [o["m"] for o in init]
    x0 = [o["p"][0] for o in init]
    y0 = [o["p"][1] for o in init]
    z0 = [o["p"][2] for o in init]
    vx0 = [o["v"][0] for o in init]
    vy0 = [o["v"][1] for o in init]
    vz0 = [o["v"][2] for o in init]

    return np.concatenate((m, x0, y0, z0, vx0, vy0, vz0))

def vector_to_planet_dict(vec):
    num_bodies = get_n_bodies(vec)
    
    planets = []
    for i in range(num_bodies):
        planets.append(
                dict(
                    m=get_m(vec, i),
                    p=(get_x(vec, i), get_y(vec, i), get_z(vec, i)),
                    v=(get_vx(vec, i), get_vy(vec, i), get_vz(vec, i)),
                )
        )

    return planets

def integrate(init, t_max, N):
    t = np.linspace(0, t_max, N)
    s0 = planet_dict_to_vector(init)
    solution = odeint(F, s0, t)
    final = vector_to_planet_dict(solution[-1])
    return final, to_positions(solution)

if __name__ == "__main__":
    t = np.linspace(0, 3600 * 24 * 365 * 2, 79)
    outcome, pos = integrate(init, t)
    anim = animate_outcome3d(
        pos, ["yellow", "blue", "red"], [30, 20, 15]
    )
    plt.show()
