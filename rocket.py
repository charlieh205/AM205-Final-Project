
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

if os.path.exists(".cache.bin"):
    sim = rebound.Simulation(".cache.bin")
else:
    sim = rebound.Simulation()
    sim.units = ('km', 'kg', 's')
    sim.add(["Sun", "Earth", "Mars"])
    sim.save(".cache.bin")

sim.move_to_com()

sun = sim.particles[0]
earth = sim.particles[1]
mars = sim.particles[2]

r_i = Quantity((earth.x, earth.y + 1e6, earth.z), km)
r_f = Quantity((58829713.0727898, 221560850.74577388, 3200329.8052330604), km)
tof = Quantity(365, day)

sols = list(lambert(poliastro.bodies.Sun.k, r_i, r_f, tof))
sol = sols[0][0]
sol = sol.value

p = rebound.Particle(
   m=earth.m / 1e6,
   x=earth.x,
   y=earth.y + 1e6,
   z=earth.z,  
   vx=float(sol[0]),
   vy=float(sol[1]),
   vz=float(sol[2]),
)
sim.add(p)

# one day
sim.dt = 24 * 60 * 60 
# one year
tmax = 365 * 24 * 60 * 60

N = 365
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
ax = plt.axes(projection='3d')

def update(i):
    ax.clear()
    ax.set_ylim(-246360812.31482115, 367235137.7813816)
    ax.set_xlim(-190289032.31830737, 227205650.0355562)
    ax.set_zlim(-7199082.133277591, 4000949.6426398293)

    for j in [1, 2, 3]:
        ax.plot(x[j, :i], y[j, :i], z[j, :i], color='black', ls='--')

    sun = ax.scatter(x[0, i], y[0, i], z[0, i], color='yellow', s=20)
    earth = ax.scatter(x[1, i], y[1, i], z[1, i], color='green', s=10)
    mars = ax.scatter(x[2, i], y[2, i], z[2, i], color='red', s=10)
    ship = ax.scatter(x[3, i], y[3, i], z[3, i], color='black', s=10)
    return earth, mars, ship

anim = FuncAnimation(fig, update, frames=range(0, x.shape[1], 5))

writervideo = PillowWriter(fps=60)
anim.save('mars.gif', writer=writervideo)
