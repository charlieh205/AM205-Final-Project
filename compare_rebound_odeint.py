import rebound
import numpy as np
import nbody as nb
import importlib
importlib.reload(nb)
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def rebound_sim():
  sim = rebound.Simulation()
  sim.units = ("AU", "kg", "day")
  sim.add(["Sun", "Earth", "Mars"], date="2018-01-01 00:00")

  sim.move_to_com()
  bodies = []
  for i, particle in enumerate(sim.particles):
    temp = {}
    temp["m"] = particle.m
    temp["p"] = (particle.x, particle.y, particle.z)
    temp["v"] = (particle.vx, particle.vy, particle.vz)
    bodies.append(temp)

  N = len(times)
  pos = np.zeros((len(sim.particles), N, 3))

  for (i, time) in enumerate(times):
    sim.integrate(time, exact_finish_time=1)

    for j in range(len(sim.particles)):
      p = sim.particles[j]
      pos[j, i, :] = np.array([p.x, p.y, p.z])

  return bodies, pos, sim.G

def plot_error(i, title, solution, pos, t):
    fig = plt.figure(figsize=(8, 6))
    errors = []
    num_bodies = 3
    for k in t:
      day_sol = np.array([solution[k, i] - solution[k, 0], solution[k, num_bodies + i] - solution[k, num_bodies],
                          solution[k, num_bodies * 2 + i] - solution[k, num_bodies * 2]])
      errors.append(np.linalg.norm(pos[i, k] - day_sol))
    plt.plot(t, errors)
    plt.title(title)
    plt.ylabel("Absolute Error AU")
    plt.xlabel("Days")

if __name__ == "__main__":
    sim_duration = 2 * 365
    times = np.arange(0, sim_duration)

    init, pos, G = rebound_sim()
    num_bodies = len(init)

    x0 = [o['p'][0] for o in init]
    y0 = [o['p'][1] for o in init]
    z0 = [o['p'][2] for o in init]
    vx0 = [o['v'][0] for o in init]
    vy0 = [o['v'][1] for o in init]
    vz0 = [o['v'][2] for o in init]
    s0 = np.concatenate((x0, y0, z0, vx0, vy0, vz0))


    solution = odeint(nb.F, s0, times, args=(G,))


    plot_error(1, "earth.png", solution, pos, times)
    plot_error(2, "mars.png", solution, pos, times)