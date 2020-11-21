import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.reload_library()
plt.rc('font', family='serif')
plt.style.use("C:\\Users\\kalni\\Anaconda3\\Lib\\site-packages\\matplotlib\\mpl-data\\stylelib\\reportstyle.mplstyle")
cs = ["#440357", "#3a538b", "#218c8d", "#44be70", "#f3e51e"]
# https://coolors.co/440357-3a538b-218c8d-44be70-f3e51e

plot = True
report = True
save = True
ms = np.array([
    1E+31,
    5E+31,
    1E+32,
    5E+32,
    1E+33,
    5E+33,
    1E+34,
])

ctimes = np.array([
    1.6754E+14,
    7.2363E+13,
    5.3558E+13,
    2.4645E+13,
    1.6547E+13,
    7.2750E+12,
    5.2417E+12,
])

stdevs = np.array([
    9.8855E+11,
    3.8077E+12,
    1.5831E+12,
    2.8189E+12,
    2.3368E+11,
    2.9279E+11,
    3.3292E+10,
])

def guess_func(m, a):
    return a * (m ** -0.5)

co_labels = ["a"]

popt, pcov = opt.curve_fit(guess_func, ms, ctimes, sigma=stdevs, p0=(1e30))
t_predict = guess_func(ms, *popt)
chisq = np.sum((ctimes - np.ravel(t_predict)) ** 2 / (np.ravel(stdevs)) ** 2)
chisq_red = chisq / (len(ctimes) - len(popt))
for i, p in enumerate(popt):
    print("{} = {}, error {}".format(co_labels[i], p, np.sqrt(pcov[i, i])))
print("reduced chisq = {}".format(chisq_red))

mfull = np.logspace(30,35,120)
"""
G = 6.67430e-11
Nfrac = (ctimes * np.log(0.11*100)/(0.138*100)) ** 2
rpredict = (G * 100 *ms * Nfrac) ** (1/3)
plt.scatter(ms, rpredict / 1e17)
plt.ylim(0,1)
plt.xscale("log")"""

if plot:
    line = guess_func(mfull, *popt)
    predict = guess_func(ms, *popt)
    massf = plt.figure()
    axm = massf.add_axes([0.15,0.38,0.8,0.5])
    ax_ = massf.add_axes([0.15,0.18,0.8,0.2])
    axm.scatter(ms, ctimes / 1e14, c=cs[0], s=8)
    axm.plot(mfull, line / 1e14, c = cs[2])
    #axm.set_yscale("log")
    ax_.axhline(y = 0, color = cs[2])
    ax_.errorbar(ms, (ctimes - predict) / predict, yerr = stdevs / predict, fmt=".", ms=5, c=cs[0], capsize=2, capthick=0.6)
    ax_.set_ylim(-0.17,0.17)
    axm.set_ylim(0,1.75)
    axm.set_yticks([0.1,0.5,1.0,1.5])
    axm.set_xlim(1e31,1e34)
    axm.set_xlim(9e30,1.1e34)
    ax_.set_xlim(9e30,1.1e34)
    axm.set_xscale("log")
    ax_.set_xscale("log")
    axm.set_xticks([])
    if report:
        axm.set_ylabel(r"Collapse Time ($\times 10^{14}$s)")
        ax_.set_xlabel("Particle Mass (kg)")
        #axm.set_title("(a)")
        if save:
            savename = "diagrams/mass_collapse.pdf"
            massf.savefig(savename, dpi=600, format="pdf")
