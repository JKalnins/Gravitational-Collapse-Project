import numpy as np
import matplotlib.pyplot as plt
from data_analyser import virial_plot

G = 6.67430e-11
# IMPORT FILE AND EXTRACT DATA/INFO
fname = "2020-03-01_15.34_rev_100p_10000s_1.0e+10dt"
report = False
data = np.load("data//{}.npz".format(fname))
print(data.files)
diffs = data["diffs"]
xt = data["xt"]
xt2 = data["xt2"]
vt = data["vt"]
vt2 = data["vt2"]
ft = data["ft"]
ft2 = data["ft2"]
kt = data["kt"]
kt2 = data["kt2"]
gt = data["gt"]
gt2 = data["gt2"]
com_times = data["com_t"]
com_times2 = data["com_t2"]
r_com_t = data["r_com_t"]
r_com_t2 = data["r_com_t2"]
npts, steps, dt, rmax, rmin, eps, m = data["info"][0:7]
npts = int(npts)
steps = int(steps)
pts = np.arange(npts)
errs = np.zeros((2,npts))

for pt in pts:
    errs[0,pt] = np.amax([xt[0,pt,0], xt2[-1,pt,0]])
    errs[1,pt] = np.amin([xt[0,pt,0], xt2[-1,pt,0]])

mass_sum = npts * m
#rhalf = rmax / (2 ** 1/3)
rhalf = rmax
trh = 0.138*(npts/np.log(0.11*npts)) * np.sqrt(rhalf ** 3 / (G * mass_sum))
print("relaxation time = {:.4} s".format(trh))

h = abs(np.amin([np.amin(xt[0,:,0]), np.amin(xt2[-1,:,0])]))
std = np.std(diffs[:,0]) / h
errs = errs / h
print("difference stdev (norm) = {:.4}".format(std))

rev = plt.figure(figsize = (8,10))
axr = rev.add_axes([0.15,0.1,0.8,0.8])
#axr.scatter(pts[0:-1:4], xt[0,0:-1:4,0] / h, s = 2, c = "r", alpha = 0.6, label="initial")
#axr.scatter(pts[0:-1:4], xt2[-1,0:-1:4,0] / h, s = 2, c = "b", alpha = 0.6, label="reversed")
#axr.errorbar(pts[1:-1:4], diffs[1:-1:4,0] / h, yerr = errs[:,1:-1:4], ls = "", c = "r", label="difference", capsize=2, marker="", alpha=0.4)
axr.scatter(pts[1:-1:4], diffs[1:-1:4,0] / h, c="g", s=5, label = "diffs")
axr.fill_between(pts, std, -1* std, color = "red", lw = 0, alpha = 0.4, label="stdev")
axr.set_xlim(0,npts - 1)
axr.set_ylim(-1.05,1.05)
axr.set_title("normalised difference in initial & reversed positions")
axr.legend(loc = "upper left")

virial_plot(kt, gt, steps, npts, m, rmax, r_com_t, dt, save = True)

if report:
    axr.set_title("(a)", size=18)
    axr.set_ylabel("Normalised Radius at $t=0$ ($t_0 / t_{0,\mathrm{max}}$)", size=18)
    #axr.set_xlabel("Particles", size=18)
    axr.tick_params(labelsize=14)
    axr.set_xticks([])
    axr.set_yticks([-1,-0.5,0,0.5,1])
    #axr.set_yticks([])
    savename = "diagrams/{}_save.pdf".format(fname[0:16])
    rev.savefig(savename, dpi=600, format="pdf")

