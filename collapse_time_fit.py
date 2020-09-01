import numpy as np
import scipy.optimize as opt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ti
from matplotlib.colors import LogNorm, Normalize

#mpl.style.reload_library()
plt.rc('font', family='serif')
plt.style.use("C:\\Users\\kalni\\Anaconda3\\Lib\\site-packages\\matplotlib\\mpl-data\\stylelib\\reportstyle.mplstyle")
cs = ["#440357", "#3a538b", "#218c8d", "#44be70", "#f3e51e"]
# https://coolors.co/440357-3a538b-218c8d-44be70-f3e51e

plot = False
hmfit = True
resplot = True
save = True

nums = np.arange(100, 1100, 100)
mass = np.logspace(31, 35, num=5, base=10)
logmass = np.log10(mass)
ext = (np.amin(nums), np.amax(nums), np.amin(logmass), np.amax(logmass))
nn, mm = np.meshgrid(nums, mass)
bignum = np.linspace(50, 1050, 110)
bigmass = np.logspace(30, 36, num=110, base=10)

times = np.swapaxes(
    np.array(
        [
            [4.7752e15, 1.4976e15, 4.856e14, 1.536e14, 4.8e13],
            [3.3376e15, 1.0328e15, 3.336e14, 1.024e14, 3.36e13],
            [2.6808e15, 8.2880e14, 2.592e14, 8.4e13, 2.64e13],
            [2.2472e15, 7.2400e14, 2.32e14, 7.28e13, 2.32e13],
            [2.0240e15, 6.3920e14, 2.04e14, 6.48e13, 2.08e13],
            [1.8568e15, 5.8480e14, 1.848e14, 5.84e13, 1.84e13],
            [1.7168e15, 5.3760e14, 1.696e14, 5.36e13, 1.68e13],
            [1.5848e15, 5.0400e14, 1.592e14, 5.12e13, 1.60e13],
            [1.4944e15, 4.7920e14, 1.488e14, 4.88e13, 1.52e13],
            [1.4200e15, 4.5040e14, 1.424e14, 4.56e13, 1.44e13],
        ]
    ),
    0,
    1,
)
t_num = np.average(times, axis=0)
std_n = np.std(times, axis=0, ddof = 1)
t_m = np.average(times, axis=1)
std_m = np.std(times, axis=1, ddof = 1)

stdevs = np.swapaxes(
    np.array(
        [
            [8.732e13, 3.240e13, 9.272e12, 2.309e12, 8.000e11],
            [3.569e13, 7.715e12, 8.776e12, 1.386e12, 4.619e11],
            [3.048e13, 9.272e12, 2.013e12, 1.222e12, 4.619e11],
            [3.812e13, 6.417e12, 1.665e12, 4.619e11, 4.000e11],
            [2.562e13, 6.110e12, 2.013e12, 4.619e11, 4.619e11],
            [9.226e12, 8.000e11, 8.000e11, 4.619e11, 4.619e11],
            [2.013e12, 3.200e12, 7.433e12, 4.619e11, 4.619e11],
            [5.619e12, 3.233e12, 5.143e12, 4.619e11, 4.000e11],
            [7.632e12, 4.105e12, 1.222e12, 4.619e11, 4.000e11],
            [2.117e12, 2.013e12, 4.619e11, 4.619e11, 4.000e11],
        ]
    ),
    0,
    1,
)


coeff = 1.6951973621752603e32
coeff_e = 2.037448537420679e29
co_labels = ["a", "m^-b", "n^-c"]


def guess_func(pt, a, b, c):
    n, m = pt  # pt is a 2xN array of N pts
    return a * (m ** (-1 * b)) * (n ** (-1 * c))
    #return a * (m ** (-1 * b)) * (n / np.log(c * n))
    #return a * (m ** (-1 * b)) * ((n ** (-1*c)) / np.log(0.11 * n))


#    return a * (m ** (-1 * b)) * (n / (np.log(n * c)))

pts = np.vstack([np.ravel(nn), np.ravel(mm)])
tflat = np.ravel(times)


# half-mass
if hmfit:
    G = 6.67430e-11
    Nfrac = (tflat * np.log(0.11*np.ravel(nn))/(0.138*np.ravel(nn))) ** 2
    rpredict = (G * np.ravel(nn) * np.ravel(mm) * Nfrac) ** (1/3) / 1e18
    def gf_N(n,a,b):
        return a * (n ** b)
    pon, pcn = opt.curve_fit(gf_N, np.ravel(nn), rpredict, p0=(3.5,-1.))
    rfit = gf_N(np.ravel(nn),*pon)
    rf_small = gf_N(nums,*pon)
    
    rp_small = np.zeros_like(rf_small)
    for i, a in enumerate(np.ravel(nn)):
        j = a // 100
        rp_small[j-1] += rpredict[i] / 5
        
    stdev2 = np.zeros_like(rf_small)
    for k, b in enumerate(np.ravel(nn)):
        j = b // 100
        stdev2[j-1] += ((rpredict[i] - rp_small[j-1]) ** 2) / 4
    
    resid = (rp_small - rf_small) / rf_small
    chisq = np.sum((rf_small - rp_small) ** 2 / (stdev2))
    chisq_red = chisq / (len(np.ravel(nn)) - len(pon))
    for i, p in enumerate(pon):
        print("{} = {}, error {}".format(i, p, np.sqrt(pcn[i, i])))
    print("reduced chisq = {}".format(chisq_red))
    hmf = plt.figure()
    axh = hmf.add_axes([0.15,0.38,0.8,0.5])
    ax_ = hmf.add_axes([0.15,0.18,0.8,0.2])
    axh.scatter(np.ravel(nn), rpredict, s=5, c = cs[0])
    ax_.errorbar(nums, resid, yerr=stdev2, fmt=".", ms=5, c=cs[0], capsize=2, capthick=0.5)
    ax_.axhline(y=0, color=cs[2])
    axh.plot(bignum, gf_N(bignum, *pon), color = cs[2])
    ax_.set_xlabel("Particle Number")
    axh.set_ylabel("Half-Mass Radius $r_h / r_{max}$")
    axh.set_xlim(80,1020)
    ax_.set_xlim(90,1010)
    axh.set_ylim(0.05,0.4)
    axh.set_yticks([0.1,0.2,0.3])
    axh.set_xticks([])
    ax_.set_ylim(-0.1,0.1)
    ax_.set_yticks([-0.05,0.05])
    if save:
        #axh.set_title("(b)")
        hmf.savefig("diagrams/hmfit.pdf", dpi=600, format="pdf")
#t_rhp = 0.138 * ((nn ** 0.5) * (1e18 ** 1.5)) / ((mm ** 0.5) * (G ** 0.5) * np.log(0.11 * nn))
#plt.scatter(nn, times / t_rhp, c = mm)
##plt.xscale("log")
#plt.colorbar()

"""
# CHECK PLOTS
fign = plt.figure()
axn = fign.add_axes((.1,.3,.8,.6))
axn_r = fign.add_axes((.1,.1,.8,.18)) # LEFT, BOTTOM, WIDTH, HEIGHT
fign.suptitle("num particles vs time")
axn.scatter(pts[0], tflat, alpha = 0.5)
axn.scatter(pts[0], guess_func(pts, *popt), marker = "x")
axn_r.scatter(pts[0], (guess_func(pts, *popt) - tflat) / guess_func(pts, *popt))
axn.set_xscale("log")
axn.set_yscale("log")
axn_r.set_xscale("log")

figm = plt.figure()
axm = figm.add_axes((.1,.3,.8,.6))
axm_r = figm.add_axes((.1,.1,.8,.18))
figm.suptitle("mass vs time")
axm.scatter(pts[1], tflat, alpha = 0.5)
axm.scatter(pts[1], guess_func(pts, *popt), marker = "x")
axm_r.scatter(pts[1], (guess_func(pts, *popt) - tflat) / guess_func(pts, *popt))
axm.set_xscale("log")
axm.set_yscale("log")
axm_r.set_xscale("log")"""


# PLOTTING DATA
if plot:
    popt, pcov = opt.curve_fit(guess_func, pts, tflat, p0=(1e30, 0.5, 2.))
    t_predict = np.reshape(guess_func(pts, *popt), np.shape(nn))
    chisq = np.sum((tflat - np.ravel(t_predict)) ** 2 / (np.ravel(stdevs)) ** 2)
    chisq_red = chisq / (len(tflat) - len(popt))
    for i, p in enumerate(popt):
        print("{} = {}, error {}".format(co_labels[i], p, np.sqrt(pcov[i, i])))
    print("reduced chisq = {}".format(chisq_red))
    bn, bm = np.meshgrid(bignum, bigmass)
    bigpts = np.vstack([np.ravel(bn), np.ravel(bm)])
    tflat_big = guess_func(bigpts, *popt)
    t_big = np.reshape(tflat_big, np.shape(bn))
    residuals = (t_predict - times) / t_predict
    res_2 = np.abs((t_predict - times)) / t_predict
    print("{} mean absolute residual".format(np.mean(res_2)))
    
    if resplot:
        fig = plt.figure(figsize=(3.4, 4))
        ax = fig.add_axes((0.2, 0.6, 0.7, 0.35))
        axr = fig.add_axes([0.2,0.1,0.7,0.35])
    else:
        fig = plt.figure()
        ax = fig.add_axes((0.2, 0.2, 0.7, 0.7))
    im = ax.pcolor(
        bn,
        bm,
        t_big,
        cmap="viridis_r",
        norm=LogNorm(vmin=tflat_big.min(), vmax=tflat_big.max()),
    )
    co = ax.contour(
        bn, bm, t_big, levels=[1e14, 1e15], colors="#FFFFFF", linestyles="dashed"
    )
    fmt = ti.LogFormatterMathtext()
    fmt.create_dummy_axis()
    plt.clabel(co, co.levels, fmt=fmt)
#    sc = ax.scatter(
#        nn,
#        mm,
#        c=residuals,
#        cmap="RdBu_r",
#        s=30,
#        edgecolor="#444444",
#        linewidth=0,
#        norm=Normalize(
#            vmin=-np.abs(residuals).max() * 1,
#            vmax=np.abs(residuals).max() * 1,
#        ),
#    )
    ax.set_xlim(90, 1010)
    ax.set_ylim(9e30, 1.1e35)
    ax.set_yscale("log")
    ax.set_title("(a)")
    ax.set_ylabel("particle mass ($kg$)")
    # ax.set_yticks([1e31, 1e33, 1e35])
    ax.set_xlabel("Number of Particles")
    ax.set_xticks([100, 500, 1000])
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Relaxation Time ($s$)")
    #cbar_resid = fig.colorbar(sc, ax=ax, orientation="vertical", ticks=[-0.03, 0, 0.03])
    #cbar.ax.tick_params(labelsize=26)
    #cbar.ax.tick_params(which = "major", length = 5, width = 2)
    #cbar.ax.tick_params(which = "minor", length = 5)
    #cbar_resid.ax.set_ylabel("normalised residuals")
    #cbar_resid.ax.tick_params(labelsize=26, length = 5, width = 2)
    #ax.tick_params(axis="both", which="major", labelsize=26, length = 5, width = 2)
    #ax.tick_params(axis="y", which="minor", length=5, width=1)
    if (not resplot) and save:
        fig.savefig("diagrams/heatmap.pdf", dpi=600, format="pdf")
    if resplot:
        res = axr.pcolor(nn,mm,residuals, cmap="RdBu_r",  norm=Normalize(vmin=-0.04,vmax=0.04))
        cres = fig.colorbar(res, ax=axr)
        axr.set_yscale("log")
        axr.set_title("(b)")
        axr.set_ylabel("particle mass ($kg$)")
        # ax.set_yticks([1e31, 1e33, 1e35])
        axr.set_xlabel("Number of Particles")
        axr.set_xticks([100, 500, 1000])
        cres.ax.set_ylabel("Normalised Residuals")
        if save:
            fig.savefig("diagrams/hm_and_errorbar.png", dpi=600, format="png")