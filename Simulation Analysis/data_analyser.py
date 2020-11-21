import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.reload_library()
plt.rc('font', family='serif')
plt.style.use("C:\\Users\\kalni\\Anaconda3\\Lib\\site-packages\\matplotlib\\mpl-data\\stylelib\\reportstyle.mplstyle")
cs = ["#440357", "#3a538b", "#218c8d", "#44be70", "#f3e51e"]
# https://coolors.co/440357-3a538b-218c8d-44be70-f3e51e

G = 6.67430e-11
# ANALYSIS FUNCTIONS AND GRAPH PLOTTING

def trh(npts, m, rmax, n_heavy = 0, m_heavy = 0):
    mass_sum = ((npts - n_heavy) * m) + n_heavy * m_heavy
    trh = 0.138*(npts/np.log(0.11*npts)) * np.sqrt((rmax / (2 ** (1/3))) ** 3 / (G * mass_sum))
    return trh


def collapse_time(kt, gt, r_com_t, detailed = False):
    """
    returns peak energy spike
    compares to time of minimum mean radius from COM
    """
    emax = np.amax(np.sum(kt, axis=1))
    emax_step = np.argmax(np.sum(kt, axis=1))
    r_com_dists = np.linalg.norm(r_com_t, axis=2)
    r_com_means = np.linalg.norm(r_com_dists, axis=1)
    rmeanmin = np.amin(r_com_means)
    rmeanmin_step = np.argmin(r_com_means)
    t_cc = (emax_step + rmeanmin_step) // 2
    if detailed:
        return t_cc, emax, rmeanmin, emax_step, rmeanmin_step
    else:
        return t_cc, emax, rmeanmin


def conservation_plot(kt, gt, steps, save = False):
    k_tot = np.sum(kt, axis=1)
    g_tot = gt
    total_e = k_tot + g_tot
    initial = total_e[0]
    con = plt.figure(figsize = (16,10))
    axc = con.add_axes([0.1,0.1,0.8,0.8])
    axc.set_title("conservation of energy plot")
    axc.plot(total_e / initial)
    axc.set_xlim([0,steps])
    #axc.set_ylim([0,1.1])


def virial_plot(kt, gt, steps, npts, m, rmax, r_com_t, dt, save=False):
    """
    plots the energy of the system over time (K, U, 2K+U)
    """
    peak_step, emax = collapse_time(kt, gt, r_com_t)[0:2]
    ptime = peak_step  * dt
    print("collapse step = {}".format(peak_step))
    print("collapse time = {:.4e} s, {:.4e} yrs".format(ptime, ptime / 31536000))
    k_norm = np.sum(kt, axis=1) / emax
    g_norm = gt / emax
    vir_norm = (2 * np.sum(kt, axis=1) + gt) / emax
    #trh_pt = trh(npts, m, rmax)
    vir = plt.figure()
    axv = vir.add_axes([0.2, 0.2, 0.7, 0.7])
    axv.plot(k_norm, color=cs[2], label = "KE") #, alpha=0.7, linewidth=3)
    axv.plot(g_norm, color=cs[3], label = "GPE") #, alpha=0.7, linewidth=3)
    axv.plot(vir_norm, color=cs[0], label="2K+U") #, linewidth=3)
    #axv.axvline(trh_pt / dt)
    axv.axvline(x=peak_step - 1, color=cs[4], linestyle="--")
    axv.set_xlabel("Steps")
    axv.set_ylabel(r"Energy ($E/E_{max}$)")
    axv.set_xlim(0, steps)
    #axv.set_xlabel("Time ($s$)", size=28)
    #axv.set_xticks([0, steps // 4, steps // 2, 3 * steps // 4, steps])
    #     axv.set_xticklabels([0,"$2\cdot 10^{15}$","$4\cdot 10^{15}$","$6\cdot 10^{15}$","$8 \cdot 10^{15}$"])
    axv.set_ylim(1.1 * np.amin(g_norm), 1.1 * np.amax(k_norm))
    #axv.set_ylim(-1.4,1.1)
    #axv.set_ylabel("$E/E_{max}$", size=28)
    #axv.set_yticks([-1, -0.5, 0, 0.5, 1])
    #axv.tick_params(axis="both", which="major", labelsize=26, length=5, width=2)
    if not save:
        axv.set_title("virial energy plot")
        axv.legend(loc = "upper left")
    else:
        vir.savefig("diagrams/virial.pdf", format="pdf", dpi=600)


def split_radii_plot(light_rct, heavy_rct, steps, cnum, cnumh, label, save=False):
    """
    plot the radii of particles in the COM frame over time
    Only plots the lowest <fraction> of masses if fraction != 1
    (should be adjusted for specific conditions i.e. if there is a range of masses)
    Inputs:
        fraction: float (must be 0<fraction<=1 to work)
    """
    rad_l = np.linalg.norm(light_rct, axis=2)
    rad_h = np.linalg.norm(heavy_rct, axis=2)
    close_l_id = np.argpartition(rad_l, kth=cnum, axis=-1)
    rad_l_c = np.take_along_axis(rad_l, close_l_id, axis=-1)[:,:cnum]
    close_h_id = np.argpartition(rad_h, kth=cnumh, axis=-1)
    rad_h_c = np.take_along_axis(rad_h, close_h_id, axis=-1)[:,:cnumh]
    mean_rl = np.mean(rad_l_c, axis=1)
    mean_rh = np.mean(rad_h_c, axis=1)
    print(np.shape(rad_l_c))
    rad = plt.figure()
    axr = rad.add_axes([0.15, 0.18, 0.8, 0.7])
    axr.set_xlim(0, steps)
    axr.plot(mean_rh, cs[0], label = "heavy")
    axr.plot(mean_rl, color=cs[2], label = "light")
    axr.set_yscale("log")
    axr.set_xlabel("Steps")
    axr.set_ylabel("Radius (m)")
    axr.set_title("({})".format(label[0]))
    if save:
            rad.savefig(
                "diagrams/split_{}.pdf".format(label[1:]), format="pdf", dpi=600
                )


def moment_of_inertia_plot(masses, steps, r_com_t, save=False):
    m_of_i = np.zeros([steps + 1])
    for i, m in enumerate(masses):
        radius = np.linalg.norm(r_com_t[:, i, :], axis=1)
        m_of_i += m * (radius ** 2)
    ine = plt.figure(figsize=(16, 10))
    axi = ine.add_axes([0.1, 0.1, 0.8, 0.8])
    axi.set_title("total moment of inertia plot")
    axi.plot(m_of_i)


if __name__ == "__main__":
    # IMPORT FILE AND EXTRACT DATA/INFO
    fname = "2020-03-11_19.34_800p_1000s_1.0e+10dt"
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
    print("{} particles".format(npts))
    # print("{} heavy, {} light".format(n_heavy, npts - n_heavy))
    # print("heavy mass = {}".format(m_heavy))
    print("light mass = {}".format(m))
    print("timestep = {}s, {} yrs".format(dt, dt / 31536000))
    trh_pt = trh(npts, m, rmax) # set n_heavy and m_heavy
    print("relaxation time = {:.4} s".format(trh_pt))
    # conservation_plot(kt, gt, steps)
    virial_plot(kt, gt, steps, npts, m, rmax, r_com_t, dt)
    #for r in [0.1,0.25,0.5,0.75,1]:
    #    split_radii_plot(npts, n_heavy, steps, r_com_t, fraction=r)