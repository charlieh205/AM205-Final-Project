import os
import copy
import rebound
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import poliastro
import poliastro.bodies
from poliastro.iod import lambert
from astropy.units import Quantity, kg, km, day, second
from matplotlib.animation import FuncAnimation, PillowWriter
import datetime as dt
from tempfile import NamedTemporaryFile

minute_s = 60
hour_s = 60 * minute_s
day_s = 24 * hour_s
year_s = 365 * day_s

if os.path.exists(".cache.bin"):
    base_sim = rebound.Simulation(".cache.bin")
else:
    base_sim = rebound.Simulation()
    base_sim.units = ("km", "kg", "s")
    base_sim.add(["Sun", "Earth", "Mars"])
    base_sim.save(".cache.bin")

base_sim.dt = day_s
base_sim.move_to_com()
base_sim.integrator = "mercurius"

sun = base_sim.particles[0]
earth = base_sim.particles[1]
mars = base_sim.particles[2]

p = rebound.Particle(
    m=earth.m / 1e6,
    x=earth.x,
    y=earth.y + 1e6,
    z=earth.z,
    vx=earth.vx,
    vy=earth.vy,
    vz=earth.vz,
)
base_sim.add(p)


def copy_sim(sim):
    with NamedTemporaryFile() as f:
        sim.save(f.name)
        new_sim = rebound.Simulation(f.name)
    return new_sim

def lambert_manuever(r_i, r_f, tof):
    sols = list(lambert(poliastro.bodies.Sun.k, r_i, r_f, tof))
    launch = sols[0][0]
    reenter = sols[0][1]
    return launch.value, reenter.value

def rocket_launch(sim, rocket_id, delay, tof, target_id):

    other_sim = copy_sim(sim)
    other_sim.integrate(delay + tof)
    target = other_sim.particles[target_id]
    target = Quantity((target.x, target.y, target.z), km)

    has_launched = False
    tof_q = Quantity(tof, second)

    def launch_fn(reb_sim):

        nonlocal has_launched
        nonlocal delay
    
        sim = reb_sim.contents
        rocket = sim.particles[rocket_id]

        if not has_launched and sim.t > delay:
            r_i = Quantity((rocket.x, rocket.y, rocket.z), km)
            r_f = target
    
            launch, _ = lambert_manuever(r_i, r_f, tof_q)

            rocket.vx = float(launch[0])
            rocket.vy = float(launch[1])
            rocket.vz = float(launch[2])

            has_launched = True

    return launch_fn

def get_trajectories(sim, times):

    N = len(times)
    pos = np.zeros((len(sim.particles), N, 3))
    
    for (i, time) in enumerate(times):
    
        sim.integrate(time, exact_finish_time=1)
    
        for j in range(len(sim.particles)):
            p = sim.particles[j]
            pos[j, i, :] = np.array([p.x, p.y, p.z])

    return pos

def animate_outcome(pos, colors, sizes):
    
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    
    fig.set_facecolor("black")
    ax.set_facecolor("black")
    ax.grid(False)
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    
    N = pos.shape[0]

    angles = np.linspace(0, 180, pos.shape[1])
    
    def update(i):
        ax.clear()
        ax.view_init(45, -1 * angles[i])
        ax.set_facecolor("black")
        ax.grid(False)
        ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    
        ax.set_ylim(-246360812.31482115, 367235137.7813816)
        ax.set_xlim(-190289032.31830737, 227205650.0355562)
        ax.set_zlim(-7199082.133277591, 4000949.6426398293)
    
        x = pos[:, :, 0]
        y = pos[:, :, 1]
        z = pos[:, :, 2]
    
        for p in range(N):
            ax.plot(x[p, :i], y[p, :i], z[p, :i], color="white", ls="--")
            ax.scatter(x[p, i], y[p, i], z[p, i], color=colors[p], s=sizes[p])
    
    anim = FuncAnimation(fig, update, frames=range(0, pos.shape[1], 5))
    return anim

if __name__ == "__main__":
    total_time = 365 * day_s
    N = 500
    times = np.linspace(0, total_time, N)
    
    for i in range(30, 300, 10):
        delay = i * day_s
        tof = total_time - delay
        sim = copy_sim(base_sim)
    
        sim.additional_forces = rocket_launch(sim, 3, delay, tof, 2)
        sim.force_is_velocity_dependent = 1
    
        pos = get_trajectories(sim, times)
        if i == 30:
            total = pos
        else:
            total = np.concatenate([total, pos[-1].reshape(1, N, 3)])
    
        np.save("total.npy", total)
