import autograd.numpy as np

import tqdm
import matplotlib.pyplot as plt

from nbody import integrate, G
from lambert import lambert

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

errs = []
fds = []
thetas = []
Ts = np.linspace(T / 10, T, 1000)
for t_max in tqdm.tqdm(Ts):
    _, p = integrate(planets, t_max, 100)
    target = p[2][-1]

    v, _ = lambert(mu, np.array([r1, 0.0, 0.0]), target, t_max, tol=1e-8)

    planets[1]["v"] = (v[0], v[1], v[2])
    outcome, p1 = integrate(planets, t_max, 100)

    flight_distance = np.linalg.norm(p1[1][0] - target)
    err = np.linalg.norm(p1[1][-1] - target)

    fds.append(flight_distance)
    errs.append(err)

    mag = np.linalg.norm(v)
    theta = np.rad2deg(np.arccos(v[0] / mag))
    thetas.append(theta)

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
