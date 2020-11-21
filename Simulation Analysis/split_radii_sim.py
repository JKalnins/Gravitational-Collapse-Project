from poisson_creator import poisson_distributor
from gravsim import timeloop
import numpy as np
from datetime import datetime as dtime

# INITIAL PARAMETERS
G = 6.67430e-11
steps = 3000
dt = 1e10
description = "_" # leave as _ or None to auto-label data

# CREATING PARTICLE LOCATIONS, MASSES, VELOCITIES
npts = 800
rmax = 1e17
rmin = ((rmax ** 3) / npts) ** (1 / 3)
eps = rmin
dims = 3
pts = poisson_distributor(npts, rmax, rmin, dims)
m = 5e32
m2 = 2*m
masses = np.full(npts, m)
nheavy = 80
for i, m in enumerate(masses):
    if i % (npts // nheavy) == 0:
        masses[i] = m2

vels = np.zeros_like(pts)

# SIMULATION
xt, vt, ft, kt, gt, com_times, r_com_t = timeloop(steps, dt, masses, pts, vels, eps)


# FILENAME CREATOR
rightnow = dtime.now()
date = rightnow.date()
hour = rightnow.hour
minute = rightnow.minute
# name is a string that will be saved with the appropriate description
# this means all files with same prefix can be called later
# we will use np.savez as the arrays will be easily re-collectable with their
# names intact
if description == None or description == "_":
    description = "{}p_{}s_{:.1e}dt".format(npts, steps, dt)
name = "{}_{}.{:02}_".format(date, hour, minute) + "{}".format(description)


# DATA SAVER
# quick & dirty method of saving relevant information to file
info = np.array([npts, steps, dt, rmax, rmin, eps, m, m2, nheavy])
print(name)
np.savez(
    "data//{}".format(name),
    xt=xt,
    vt=vt,
    ft=ft,
    kt=kt,
    gt=gt,
    com_t=com_times,
    r_com_t=r_com_t,
    masses = masses,
    info = info
)