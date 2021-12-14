
import rebound
from utils import get_trajectories
import numpy as np
import matplotlib.pyplot as plt
import plot as splt
from plot import animate_outcome2d, plot_trajectory, init_axes, plot_trajectories
from utils import *
from nbody import integrate, G

def hohman(mu, r1, r2):

    # Orbit exit delta v
    delta_v1 = np.sqrt(mu / r1) * ( np.sqrt( ( 2 * r2) / (r1 + r2)) - 1)

    # Orbit re-entry delta v
    delta_v2 = np.sqrt(mu / r2) * ( 1 - np.sqrt( ( 2 * r1) / (r1 + r2)))

    # Time of manuever
    t_h = np.pi * np.sqrt((r1 + r2)**3 / (8 * mu))

    return t_h, delta_v1, delta_v2

primary = dict(
    m=1.0/G,
    p=(0.0, 0.0, 0.0),
    v=(0.0, 0.0, 0.0),
)

smaller_orbit = dict(
    m=0.0,
    p=(1.0, 0.0, 0.0),
    v=(0.0, 1.0, 0.0),
)

larger_orbit = dict(
    m=0.0,
    p=(2.0, 0.0, 0.0),
    v=(0.0, np.sqrt(1.0/2.0), 0.0),
)

planets = [
    primary,
    larger_orbit, 
    smaller_orbit,
]

t_h, delta_v1, delta_v2 = hohman(1.0, 1.0, 2.0)

outcome, p1 = integrate(planets, 2 * np.pi, 100)

outcome[2]['v'] = (0.0, outcome[2]['v'][1] + delta_v1, 0.0)
outcome, p2 = integrate(outcome, t_h, 100)

outcome[2]['v'] = (0.0, outcome[2]['v'][1] - delta_v2, 0.0)
outcome, p3 = integrate(outcome, np.pi, 100)

pos = np.concatenate([p1, p2, p3], axis=1)
anim = animate_outcome2d(pos, ["yellow", "green", "red"], [100, 30, 30], skip=5)
plt.show()
