import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


class point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class body:
    def __init__(self, location, mass, velocity, name=""):
        self.location = location
        self.mass = mass
        self.velocity = velocity
        self.name = name


def calculate_single_body_acceleration(bodies, body_index):
    G_const = 6.67408e-20
    acceleration = point(0, 0, 0)
    target_body = bodies[body_index]
    for index, external_body in enumerate(bodies):
        if index != body_index:
            r = (target_body.location.x - external_body.location.x) ** 2 + (
                        target_body.location.y - external_body.location.y) ** 2 + (
                            target_body.location.z - external_body.location.z) ** 2
            r = math.sqrt(r)
            tmp = G_const * external_body.mass / r ** 3
            acceleration.x += tmp * (external_body.location.x - target_body.location.x)
            acceleration.y += tmp * (external_body.location.y - target_body.location.y)
            acceleration.z += tmp * (external_body.location.z - target_body.location.z)

    return acceleration


def compute_velocity(bodies, time_step=1):
    for body_index, target_body in enumerate(bodies):
        acceleration = calculate_single_body_acceleration(bodies, body_index)

        target_body.velocity.x += acceleration.x * time_step
        target_body.velocity.y += acceleration.y * time_step
        target_body.velocity.z += acceleration.z * time_step


def update_location(bodies, time_step=1):
    for target_body in bodies:
        target_body.location.x += target_body.velocity.x * time_step
        target_body.location.y += target_body.velocity.y * time_step
        target_body.location.z += target_body.velocity.z * time_step


def compute_gravity_step(bodies, time_step=1):
    compute_velocity(bodies, time_step=time_step)
    update_location(bodies, time_step=time_step)


def run_simulation(bodies, names=None, time_step=1, number_of_steps=10000, report_freq=100):
    # create output container for each body
    body_locations_hist = []
    for current_body in bodies:
        body_locations_hist.append({"x": [], "y": [], "z": [], "name": current_body.name})

    for i in range(1, number_of_steps):
        compute_gravity_step(bodies, time_step=1000)

        if i % report_freq == 0:
            for index, body_location in enumerate(body_locations_hist):
                body_location["x"].append(bodies[index].location.x)
                body_location["y"].append(bodies[index].location.y)
                body_location["z"].append(bodies[index].location.z)

    return body_locations_hist


def animate_outcome(pos, colors, sizes):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    fig.set_facecolor("black")
    ax.set_facecolor("black")
    ax.grid(False)
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    N = len(pos)


    def update(i):
        ax.clear()
        ax.set_facecolor("black")
        ax.grid(False)
        ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

        ax.set_ylim(-246360812.31482115, 6e8)
        ax.set_xlim(-190289032.31830737, 227205650.0355562)
        ax.set_zlim(-7199082.133277591, 4000949.6426398293)


        for p in range(N):
            x = pos[p]['x']
            y = pos[p]['y']
            z = pos[p]['z']
            ax.plot(x[:i], y[:i], z[:i], color="white", ls="--")
            ax.scatter(x[i], y[i], z[i], color=colors[p], s=sizes[p])

    anim = FuncAnimation(fig, update, frames= len(pos[0]['x']))
    return anim

sun = {
    "mass": 1.9884754159566474e+30,
    "location": point(130.12022433639504, 493.88921553478576, 1.9894885155190423),
    "velocity": point(-9.08252417469984e-05, 2.763470714263038e-05, 9.333023252795563e-08)
}
earth = {
    "mass": 6.045825576341311e+24,
    "location": point(-52532397.16477036, -142290670.7988091, 6714.9380375893525),
    "velocity": point(27.459556210605758, -10.428715463342813, 0.0004283693350210454)
}

mars = {
    "mass": 6.417120534329241e+23,
    "location": point(91724696.20692892, -189839018.6923888, -6228099.232650615),
    "velocity": point(22.733051422552098, 12.621328917003236, -0.29323856116219776)
}

if __name__ == "__main__":
    # build list of planets in the simulation, or create your own
    bodies = [
        body(location=sun["location"], mass=sun["mass"], velocity=sun["velocity"], name="sun"),
        body(location=earth["location"], mass=earth["mass"], velocity=earth["velocity"], name="earth"),
        body(location=mars["location"], mass=mars["mass"], velocity=mars["velocity"], name="mars"),
    ]

    pos = run_simulation(bodies, time_step=100, number_of_steps=80000, report_freq=1000)
    anim = animate_outcome(pos, ["yellow", "blue" , "red"], [30,20,15])
    anim.save("nbody.gif", writer="imagemagick", fps=2)

