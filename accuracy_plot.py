import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib as mpl

mpl.style.reload_library()
plt.rc('font', family='serif')
plt.style.use("C:\\Users\\kalni\\Anaconda3\\Lib\\site-packages\\matplotlib\\mpl-data\\stylelib\\reportstyle.mplstyle")
cs = ["#440357", "#3a538b", "#218c8d", "#44be70", "#f3e51e"]
# https://coolors.co/440357-3a538b-218c8d-44be70-f3e51e

save = True

diffs = np.array([
    [0.006166, 0.007388, 0.008023],
    [0.008426, 0.007969, 0.005306],
    [0.02778, 0.02493, 0.01656],
    [0.02636, 0.03803, 0.06360],
    [0.05102, 0.05524, 0.07527],
    [0.1709, 0.2045, 0.1902],
    ]
)

nsteps = np.array([
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    ]
)
nbig = np.arange(0,7000, step=100)

diffmeans = np.mean(diffs, axis=1)
    
diffstd = np.std(diffs, axis=1, ddof=1)

def guess_func(n, a, b):
    return a * np.exp(b * n)

co_labels = ["a", "b"]

popt, pcov = opt.curve_fit(guess_func, nsteps, diffmeans, sigma=diffstd, p0=(0.001,0.0009))
diffit = guess_func(nsteps, *popt)
chisq = np.sum((diffmeans - diffit) ** 2 / (diffstd ** 2))
chisq_red = chisq / (len(diffmeans) - len(popt))
for i, p in enumerate(popt):
    print("{} = {}, error {}".format(co_labels[i], p, np.sqrt(pcov[i, i])))
print("reduced chisq = {}".format(chisq_red))

fig = plt.figure()
axm = fig.add_axes([0.15,0.38,0.8,0.5])
ax_ = fig.add_axes([0.15,0.18,0.8,0.2])

axm.scatter(nsteps, diffmeans, c=cs[0], s=10)
axm.plot(nbig, guess_func(nbig, *popt), c=cs[2])
ax_.axhline(y = 0, color = cs[2])
ax_.errorbar(nsteps, (diffmeans - diffit) / diffit, yerr = diffstd / diffit, fmt=".", ms=5, c=cs[0], capsize=2, capthick=0.6)

axm.set_ylim(0,0.2)
axm.set_xlim(900,6100)
ax_.set_xlim(900,6100)
axm.set_xticks([])
ax_.set_yticks([-0.3,0.3])
axm.set_ylabel("Mean Position Error")
ax_.set_xlabel("Number of Steps")

if save:
    savename = "diagrams/accuracy_plot.pdf"
    fig.savefig(savename, dpi=600, format="pdf")
