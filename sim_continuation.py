import numpy as np
from gravsim import timeloop
from datetime import datetime as dtime

fname = "2020-03-07_14.30_100p_40000s_1.0e+10dtx2"
data = np.load("data//{}.npz".format(fname))
print(data.files)
xt = data["xt"]
vt = data["vt"]
ft = data["ft"]
kt = data["kt"]
gt = data["gt"]
com_times = data["com_t"]
r_com_t = data["r_com_t"]
# npts, n_heavy, steps, dt, rmax, rmin, eps, m, m_heavy = data["info"][0:9]
npts, steps, dt, rmax, rmin, eps, m = data["info"][0:7]
npts = int(npts)
#n_heavy = int(n_heavy)
steps = int(steps)

pts = xt[-1,:,:]
masses = np.full((npts), m)
vels = vt[-1,:,:]

xt2, vt2, ft2, kt2, gt2, com_times2, r_com_t2 = timeloop(steps, dt, masses, pts, vels, eps)

rightnow = dtime.now()
date = rightnow.date()
hour = rightnow.hour
minute = rightnow.minute

# name is a string that will be saved with the appropriate description
# this means all files with same prefix can be called later
# we will use np.savez as the arrays will be easily re-collectable with their
# names intact

name = "{}x2".format(fname)


# DATA SAVER
# quick & dirty method of saving relevant information to file
info2 = np.array([npts, steps, dt, rmax, rmin, eps, m])
print(name)
np.savez(
    "data//{}".format(name),
    xt=xt2,
    vt=vt2,
    ft=ft2,
    kt=kt2,
    gt=gt2,
    com_t=com_times2,
    r_com_t=r_com_t2,
    masses = masses,
    info = info2
)