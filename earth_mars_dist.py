import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import rebound
import math
from tqdm.auto import tqdm as tqdm_auto
from os.path import exists


#############################
### HELPER FUNCTIONS      ###
#############################
def mjd_to_jd(mjd):
    return mjd + 2400000.5

def jd_to_mjd(jd):
    return jd - 2400000.5

def date_to_jd(year, month, day):
    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month
    if ((year < 1582) or (year == 1582 and month < 10) or (year == 1582 and month == 10 and day < 15)):
        B = 0
    else:
        A = math.trunc(yearp / 100.)
        B = 2 - A + math.trunc(A / 4.)
    if yearp < 0:
        C = math.trunc((365.25 * yearp) - 0.75)
    else:
        C = math.trunc(365.25 * yearp)
    D = math.trunc(30.6001 * (monthp + 1))
    jd = B + C + D + day + 1720994.5
    return jd

def date_to_mjd(year, month, day):
    jd = date_to_jd(year, month, day)
    mjd = jd_to_mjd(jd)
    return mjd

def jd_to_date(jd):
    jd += 0.5
    F, I = math.modf(jd)
    I = int(I)
    A = math.trunc((I - 1867216.25)/36524.25)
    if I > 2299160:
        B = I + 1 + A - math.trunc(A / 4.)
    else:
        B = I
    C = B + 1524
    D = math.trunc((C - 122.1) / 365.25)
    E = math.trunc(365.25 * D)
    G = math.trunc((C - E) / 30.6001)
    day = C - E + F - math.trunc(30.6001 * G)
    if G < 13.5:
        month = G - 1
    else:
        month = G - 13
    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715
    return year, month, day

def mjd_to_date(mjd):
    jd = mjd_to_jd(mjd)
    year, month, day = jd_to_date(jd)
    return year, month, day

def body_slice(data, name_dict, short_name):
    r = name_dict[short_name]
    j0 = 3 * r
    j1 = j0 + 3
    return data[:, j0:j1]

def au_to_km(au):
    return 149597871 * au

#############################
### SET UP THE SIMULATION ###
#############################
# get today's date
TODAY = dt.date.today()
TODAY_STR = TODAY.strftime("%Y-%m-%d 00:00")
TODAY_MJD = date_to_mjd(*[int(n) for n in TODAY_STR.split(" ")[0].split("-")])
# define simulation objects
body_names = ["Sun", "Earth", "Mars Barycenter"]
horizon_names = body_names
body_ids = [10, 399, 400]
body_id_strs = [str(id) for id in body_ids]
short_names = [name.split()[0] for name in body_names]
name_to_idx = {k: v for v, k in enumerate(short_names)}
# try to load in existing simulation
if exists(f"planets_{int(TODAY_MJD)}.bin"):
    sim = rebound.Simulation(f"planets_{int(TODAY_MJD)}.bin")
else:
    # create empty simulation
    sim = rebound.Simulation()
    # set preferred units
    sim.units = ("day", "AU", "Msun")
    # add Sun, Earth, and Mars to simulation at today's date
    for name in horizon_names:
        sim.add(name, date=TODAY_STR)
    sim.t = TODAY_MJD
    sim.dt = 1. / 16.
    sim.save(filename=f"planets_{int(TODAY_MJD)}.bin")

#############################
### SIMULATE 10 YEARS     ###
#############################
# set number of years
n_years = 10
# set number of days (integer)
M = np.round(n_years * 365.25).astype(np.int32) + 1
# number of particles in simulation
N = np.int32(sim.N)
# allocate position and velocity arrays
q = np.zeros((M, 3 * N), dtype=np.float64)
v = np.zeros((M, 3 * N), dtype=np.float64)
# create array of times
ts = np.arange(TODAY_MJD, TODAY_MJD + M)
# integrate simulation and save state vectors
idx = tqdm_auto(list(enumerate(ts)))
for i, t in idx:
    # integreate to current time step
    sim.integrate(t, exact_finish_time = 1)
    # save position
    sim.serialize_particle_data(xyz = q[i])
    # save velocity
    sim.serialize_particle_data(vxvyvz = v[i])

#############################
### VISUALIZE RESULTS     ###
#############################
# unpack position of Earth and Mars
q_earth = body_slice(q, name_to_idx, "Earth")
q_mars = body_slice(q, name_to_idx, "Mars")
# distance from Earth to Mars
r_mars = np.linalg.norm(q_mars - q_earth, axis = 1)
# set number of xticks
n_xticks = 20
xticks = [int(x) for x in np.linspace(ts[0], ts[-1], num = n_xticks)]
xtick_labs = [d.split(".")[0] for d in ["-".join([str(a) for a in mjd_to_date(x)]) for x in xticks]]
# plot distance from Earth to Mars
plt.figure(figsize = (16, 10))
plt.plot(ts, r_mars)
ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xtick_labs, rotation = 90)
plt.xlabel("Date", fontsize = 18)
plt.ylabel("Distance (AU)", fontsize = 18)
plt.title("Distance between Earth and Mars", fontsize = 24)
plt.grid()
plt.tight_layout()
plt.savefig("distance_earth_mars.png")