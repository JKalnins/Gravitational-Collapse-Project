import numpy as np
from data_analyser import virial_plot

flist = [
    "2020-03-07_14.30_100p_40000s_1.0e+10dt",
    "2020-03-07_14.30_100p_40000s_1.0e+10dtx2",
    "2020-03-07_14.30_100p_40000s_1.0e+10dtx2x2",
]

init = np.load("data//{}.npz".format(flist[0]))
npts, steps, dt, rmax, rmin, eps, m = init["info"][0:7]
npts = int(npts)
steps = int(steps)

#long_kt = np.empty((steps,npts))
#long_gt = np.empty((steps))
#long_rct = np.empty((steps, npts, 3))
#l_steps = 0
for i, fn in enumerate(flist):
    data = np.load("data//{}.npz".format(fn))
    if i == 0:
        long_kt = data["kt"]
        long_gt = data["gt"]
        long_rct = data["r_com_t"]
        l_steps = int(steps)
    else:
    #    print(data.files)
    #    xt = data["xt"]
    #    vt = data["vt"]
    #    ft = data["ft"]
        long_kt = np.append(long_kt, data["kt"], axis = 0)
        long_gt = np.append(long_gt, data["gt"], axis = 0)
    #    com_times = data["com_t"]
        long_rct = np.append(long_rct, data["r_com_t"], axis = 0)
        l_steps += int(steps)

virial_plot(long_kt, long_gt, l_steps, npts, m, rmax, long_rct, dt)