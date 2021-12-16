import os
from dataclasses import dataclass

import rebound
from collections import deque
from tempfile import NamedTemporaryFile
import autograd.numpy as np


def get_solar_system(planets, date):

    date_str = date.strftime("%Y-%m-%d 12:00")
    os.makedirs(".cache", exist_ok=True)

    cache_name = f".cache/{date_str}.bin"
    if os.path.exists(cache_name):
        sim = rebound.Simulation(cache_name)
    else:

        sim = rebound.Simulation()
        sim.units = ("kg", "AU", "s")
        sim.add(planets, date=date_str)
        sim.move_to_com()
        sim.save(cache_name)

    init = []
    for (i, p) in enumerate(sim.particles):

        if i == 0:
            loc = (0, 0, 0)
        else:
            primary = sim.particles[0]
            loc = (p.x - primary.x, p.y - primary.y, p.z - primary.z)

        init.append(
            {
                "m": p.m,
                "p": (p.x, p.y, p.z),
                "v": (p.vx, p.vy, p.vz),
            }
        )
    return init, sim.G


def copy_sim(sim):
    with NamedTemporaryFile() as f:
        sim.save(f.name)
        new_sim = rebound.Simulation(f.name)
    return new_sim


def get_trajectories(sim, times):

    N = len(times)
    pos = np.zeros((len(sim.particles), N, 3))

    for (i, time) in enumerate(times):

        sim.integrate(time, exact_finish_time=1)

        for j in range(len(sim.particles)):
            p = sim.particles[j]
            pos[j, i, :] = np.array([p.x, p.y, p.z])

    return pos


class UnionPulse:
    def __init__(self, pulses):
        self.pulses = pulses

    def apply(self, reb_sim):
        for p in self.pulses:
            p.apply(reb_sim)

    def __call__(self, reb_sim):
        return self.apply(reb_sim)


def apply_velocity_vec(particle, velocity):
    particle.vx = velocity[0]
    particle.vy = velocity[1]
    particle.vz = velocity[2]


def get_velocity_vec(particle):
    return np.array([particle.vx, particle.vy, particle.vz])


def get_position_vec(particle):
    return np.array([particle.x, particle.y, particle.z])


class Pulse:
    def __init__(self, when_fn, apply_fn):
        self.when_fn = when_fn
        self.apply_fn = apply_fn

        self.attempt_history = deque(maxlen=10)
        self.done = False
        self.tol = 1e-3

    def apply(self, reb_sim):

        if self.done:
            return

        sim = reb_sim.contents

        if len(self.attempt_history) > 0:
            if (
                np.linalg.norm(
                    get_velocity_vec(sim.particles[1]) - self.attempt_history[-1], 2
                )
                < self.tol
            ):
                self.done = True
                return

        if self.when_fn(sim):
            desired_velocity = self.apply_fn(sim.particles[1])
            self.attempt_history.append(desired_velocity)
            apply_velocity_vec(sim.particles[1], desired_velocity)

    def __call__(self, reb_sim):
        return self.apply(reb_sim)

    def __add__(self, other):
        return UnionPulse([self, other])

    def __ladd__(self, other):
        return UnionPulse([self, other])

    def __radd__(self, other):
        return UnionPulse([self, other])
