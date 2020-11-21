import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from data_analyser import split_radii_plot, virial_plot

mpl.style.reload_library()
plt.rc('font', family='serif')
plt.style.use("C:\\Users\\kalni\\Anaconda3\\Lib\\site-packages\\matplotlib\\mpl-data\\stylelib\\reportstyle.mplstyle")
cs = ["#440357", "#3a538b", "#218c8d", "#44be70", "#f3e51e"]
# https://coolors.co/440357-3a538b-218c8d-44be70-f3e51e


#fname = "2020-03-12_10.24_800p_1000s_1.0e+10dt"
#fname = "2020-03-12_10.29_800p_1000s_1.0e+10dt"
#fname = "2020-03-12_10.39_800p_2000s_1.0e+10dt" # s=0.1415 20/800
#fname = "2020-03-12_10.50_800p_2000s_1.0e+10dt" # s=0.0716 10/800
#fname = "2020-03-12_10.59_800p_2000s_1.0e+10dt" # s=0.298 40/800
fname = "2020-03-12_11.59_800p_3000s_1.0e+10dt" # s=0.629 80/800
data = np.load("data//{}.npz".format(fname))
print(data.files)
xt = data["xt"]
vt = data["vt"]
ft = data["ft"]
kt = data["kt"]
gt = data["gt"]
com_times = data["com_t"]
r_com_t = data["r_com_t"]
masses = data["masses"]
npts, steps, dt, rmax, rmin, eps, m1, m2, nheavy = data["info"]

npts = int(npts)
steps = int(steps)
nheavy = int(nheavy)
nlight = npts - nheavy

mfr = (m2/m1) ** 1.5
nfr = (m2*nheavy/(m1*nlight))
print("Spitzer Value = {:.3}".format(mfr * nfr))

light_rct = np.zeros((steps+1, nlight, 3))
heavy_rct = np.zeros((steps+1, nheavy, 3))
a = 0
b = 0
for i, m in enumerate(masses):
    
    if m == m1:
        light_rct[:,a,:] = r_com_t[:,i,:]
        a += 1
    else:
        heavy_rct[:,b,:] = r_com_t[:,i,:]
        b += 1

#virial_plot(kt, gt, steps, npts, m, rmax, r_com_t, dt, save=False)
split_radii_plot(light_rct, heavy_rct, steps, cnum = 40, cnumh = 4, label = "bheavy", save=false)
