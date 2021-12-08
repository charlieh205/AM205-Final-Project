import os
import copy
import rebound
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import poliastro
import poliastro.bodies
from poliastro.iod import lambert
from astropy.units import Quantity, kg, km, day
from matplotlib.animation import FuncAnimation, PillowWriter
import datetime as dt

minute_s = 60
hour_s = 60 * minute_s
day_s = 24 * hour_s
year_s = 365 * day_s


def get_trajectories(r_i, r_f, tof):
    sols = list(lambert(poliastro.bodies.Sun.k, r_i, r_f, tof))
    sol = sols[0][0]
    return sol.value


if os.path.exists(".cache.bin"):
    sim = rebound.Simulation(".cache.bin")
else:
    sim = rebound.Simulation()
    sim.units = ("km", "kg", "s")
    sim.add(["Sun", "Earth", "Mars"])
    sim.save(".cache.bin")

sim.move_to_com()

sun = sim.particles[0]
earth = sim.particles[1]
mars = sim.particles[2]

p = rebound.Particle(
    m=earth.m / 1e6,
    x=earth.x,
    y=earth.y + 1e6,
    z=earth.z,
    vx=earth.vx,
    vy=earth.vy,
    vz=earth.vz,
)
sim.add(p)

rocket = sim.particles[-1]

delay = 30 * day_s
has_launched = False


def rocket_launch(reb_sim):
    global has_launched
    global delay

    sim = reb_sim.contents
    if not has_launched and sim.t > delay:
        r_i = Quantity((rocket.x, rocket.y, rocket.z), km)
        r_f = Quantity((45822807.23067759, 225907296.33619067, 3610478.5898426753), km)
        tof = Quantity(345, day)

        sol = get_trajectories(r_i, r_f, tof)
        rocket.vx = float(sol[0])
        rocket.vy = float(sol[1])
        rocket.vz = float(sol[2])
        has_launched = True


sim.additional_forces = rocket_launch
sim.force_is_velocity_dependent = 1

sim.dt = day_s
N = 375
tmax = N * day_s
times = np.linspace(0, tmax, N)
sim.integrator = "mercurius"

x = np.zeros((len(sim.particles), N))
y = np.zeros((len(sim.particles), N))
z = np.zeros((len(sim.particles), N))

for (i, time) in enumerate(times):

    sim.integrate(time, exact_finish_time=1)

    for j in range(len(sim.particles)):
        x[j, i] = sim.particles[j].x
        y[j, i] = sim.particles[j].y
        z[j, i] = sim.particles[j].z

fig = plt.figure()
ax = plt.axes(projection="3d")

fig.set_facecolor("black")
ax.set_facecolor("black")
ax.grid(False)
ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))


def update(i):
    ax.clear()
    ax.set_facecolor("black")
    ax.grid(False)
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    ax.set_ylim(-246360812.31482115, 367235137.7813816)
    ax.set_xlim(-190289032.31830737, 227205650.0355562)
    ax.set_zlim(-7199082.133277591, 4000949.6426398293)

    for j in [1, 2, 3]:
        ax.plot(x[j, :i], y[j, :i], z[j, :i], color="white", ls="--")

    sun = ax.scatter(x[0, i], y[0, i], z[0, i], color="yellow", s=100)
    earth = ax.scatter(x[1, i], y[1, i], z[1, i], color="green", s=30)
    mars = ax.scatter(x[2, i], y[2, i], z[2, i], color="red", s=30)
    ship = ax.scatter(x[3, i], y[3, i], z[3, i], color="silver", s=10)
    return earth, mars, ship


anim = FuncAnimation(fig, update, frames=range(0, x.shape[1], 5))
plt.show()
