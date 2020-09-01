from poisson_creator import poisson_distributor
from gravsim import timeloop
import numpy as np
from datetime import datetime as dtime

# INITIAL PARAMETERS
steps = 6000
dt = 5e9
description = "_" # leave as _ or None to auto-label data

# CREATING PARTICLE LOCATIONS, MASSES, VELOCITIES
npts = 100
rmax = 1e17
rmin = ((rmax ** 3) / npts) ** (1 / 3)
eps = rmin
dims = 3
pts = poisson_distributor(npts, rmax, rmin, dims)
m = 1e34
masses = np.full(npts, m)
vels = np.zeros_like(pts)
info = np.zeros([9])
print("npts = {}, {} steps, particle mass = {}".format(npts, steps, m))

# SIMULATION
xt, vt, ft, kt, gt, com_times, r_com_t = timeloop(steps, dt, masses, pts, vels, eps)
print("forwards done")

# REVERSING
pts2 = xt[-1,:,:]
vels2 = vt[-1,:,:]
dt2 = -1 * dt

# SIMULATION BACKWARDS
xt2, vt2, ft2, kt2, gt2, com_times2, r_com_t2 = timeloop(steps, dt2, masses, pts2, vels2, eps)
print("backwards done")

# DIFFERENCE
diffs = xt2[-1,:,:] - pts

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
    description = "rev_{}p_{}s_{:.1e}dt".format(npts, steps, dt)
name = "{}_{}.{:02}_".format(date, hour, minute) + "{}".format(description)


# DATA SAVER
# quick & dirty method of saving relevant information to file
info[0:7] = npts, steps, dt, rmax, rmin, eps, m
print(name)
np.savez(
    "data//{}".format(name),
    diffs=diffs,
    xt=xt,
    xt2=xt2,
    vt=vt,
    vt2=vt2,
    ft=ft,
    ft2=ft2,
    kt=kt,
    kt2=kt,
    gt=gt,
    gt2=gt2,
    com_t=com_times,
    com_t2=com_times2,
    r_com_t=r_com_t,
    r_com_t2=r_com_t2,
    masses = masses,
    info = info
)

