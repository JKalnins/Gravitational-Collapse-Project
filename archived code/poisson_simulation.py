import numpy as np
from poisson_creator import poisson_distributor
from gravsim import timeloop
import matplotlib.pyplot as plt
from numpy import sin, cos
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from datetime import datetime as dtime


np.random.seed(1998)
plt.style.use("seaborn-dark")
parsec = 3.08567758149137e16


class idiot(Exception):  # this is very useful
    pass


n = 1000
rmax = 1e18
dmin = 1e17  # minimum dist between pts
dims = 3  # no of dimensions 2 = circle, 3 = sphere
omega = 0
vmax = 0.0  # currently set so vx = vy = vz = vmax for all particles
steps = 2000
nframes = 200
dt = 5e12  # in seconds
m = np.full((n), 1e30)
vs = np.full((n, 3), vmax)
eps = dmin / 10
animate = False
energy_plot = True
xy_plot = True
xz_plot = False
save_ani = False
save = False

rightnow = dtime.now()
date = rightnow.date()
hour = rightnow.hour
minute = rightnow.minute

name = "figs//poisson_{}_{}.{:02}_".format(date, hour, minute) + "{}.{}"
if save:
    with open(name.format("info","txt"),"w+") as out:
        out.writelines(
                ["{} particles\n".format(n),
                 "{}  steps\n".format(steps),
                 "{:.2} pc radius\n".format(rmax / parsec),
                 "{:.2} pc min separation\n".format(dmin / parsec)]
        )

if animate and steps < nframes:
    raise idiot

# Creating a Poisson sphere - all pts must be at least mindist from all others

pts = poisson_distributor(n, rmax, dmin, dims)


# changing particles
for i in range(pts.shape[0]):
    obj = pts[i, :]
    r = np.sqrt(np.sum(obj ** 2))
    theta = np.arctan(obj[1] / obj[0])
    if obj[0] < 0:
        theta += np.pi
    theta_v = theta + (np.pi / 2)  # direction of v is perp to r for circular motion
    v = omega * r
    vs[i, 0:2] = v * cos(theta_v), v * sin(theta_v)
colour = ["green" if i != -1 else "black" for i in range(n)]
size = [4 if i != -1 else 8 for i in range(n)]

# TIMELOOP

print(
    " {} particles\n".format(n),
    "{}  steps\n".format(steps),
    "{:.2} pc radius\n".format(rmax / parsec),
    "{:.2} pc min separation\n".format(dmin / parsec),
)

xt, vt, ft, kt, gt, com_times, r_com_t = timeloop(steps, dt, m, pts, vs, eps)


# GRAPHS & STATS

# finding particles that are still in cluster
in_radius = [[] for i in range(steps + 1)]
rad_coms = np.empty_like(kt)
sum_kt_in = np.zeros_like(gt)
rad_com_means = np.zeros_like(gt)   
for i in range(steps + 1):
    rad_coms[i, :] = np.linalg.norm(r_com_t[i, :, :], axis = 1)
    rad_com_means[i] = np.linalg.norm(rad_coms[i, :])
    for j, r in enumerate(rad_coms[i, :]):
        if r > 0 * rad_com_means[i]: # currently not removing any values
            in_radius[i].append(j)
            sum_kt_in[i] += kt[i, j]
# creating circle for initial radius
th = np.linspace(0, 2 * np.pi, 100)
tlabels = np.linspace(0, steps, num=6, dtype = int)
times_labels = tlabels * dt

labels = ["step {}", "step {}", "step {}", "step {}", "step {}", "step {}"]
avg_dist = []
mean_k_in = np.mean(sum_kt_in)
sum_kt_all = np.sum(kt, axis = 1)
mean_kt_all = np.mean(sum_kt_all)
mean_g = np.mean(gt)
ts = np.linspace(0, steps * dt, num=steps + 1)
elabels = ["K", "U", "K + U", "2K + U"]
label_colours = ["green", "orange", "blue", "red"]


# Animation
if animate:
    fig_an, ax_an = plt.subplots()
    ax_an.set_xlim(-2 * rmax, 2 * rmax)
    ax_an.set_ylim(-2 * rmax, 2 * rmax)
    plot = ax_an.plot(rmax * cos(th), rmax * sin(th), lw=1, color="red", alpha=0.2)
    scatter = ax_an.scatter(xt[0, :, 0], xt[0, :, 1], s=size, c=colour)
    frames = np.linspace(0, steps, num=nframes, dtype=int)


def update_an(i):
    """updates frame in animation"""
    scatter.set_offsets(xt[frames[i], :, 0:2])
    annotation = ax_an.annotate(
        "step {} of {}".format(i, len(frames)),
        xy=(0.05, 0.05),
        xycoords="axes fraction",
        size=8,
    )
    return (scatter, annotation)


if animate:
    """animates positions over time based on update_an"""
    ani = FuncAnimation(
        fig_an,
        update_an,
        interval=30,
        frames=nframes,
        blit=True,
        save_count=nframes,
        repeat=True,
    )
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    fig_an.show()
    if save_ani:
        ani.save(name.format("animation", "gif"), writer=writer)


# energy
if energy_plot:
    fige, axe = plt.subplots(1, 1, sharex=True)
    axe.plot([0, steps], [0, 0], color='#000000')
    axe.plot(sum_kt_in, color=label_colours[0])
    axe.plot(gt, color=label_colours[1])
    axe.plot(gt + sum_kt_in, color=label_colours[2])
    axe.plot(gt + 2 * sum_kt_in, color=label_colours[3])
    axe.plot([np.argmax(sum_kt_in), np.argmax(sum_kt_in)],[-1e50,1e50])
    axe.set_xlim(0, steps)
    axe.set_xlabel("steps")
    axe.set_ylim(np.amin(gt), - np.amin(gt))  
    axe.set_ylabel("Energy [J]")
    for k, label in enumerate(elabels):
        for r in range(6):
            axe.plot(
                [times_labels[r], times_labels[r]],
                [-1e10, 1e10],
                color="red",
                linewidth=1,
            )
        axe.annotate(
            label,
            xy=(0.05 + 0.25 * k, 0.95),
            xycoords="axes fraction",
            size=10,
            color=label_colours[k],
        )
    fige.tight_layout()
    if save:
        fige.savefig(name.format("energy", "png"))

# XY-scatter plot
if xy_plot:
    figx, axx = plt.subplots(2, 3)
    for i, t in enumerate(tlabels):
        rads = np.linalg.norm(xt[t, :, :], axis = 1)
        r_mean = np.mean(rads)
        avg_dist.append("<r> = {:.3}".format(r_mean))
        if i < 3:  # top row
            axx[0, i].set_xlim(-2.5 * r_mean, 2.5 * r_mean)
            axx[0, i].set_ylim(-2.5 * r_mean, 2.5 * r_mean)
            axx[0, i].set_aspect("equal")
            axx[0, i].annotate(avg_dist[i], xy=(0.05, 0.93), xycoords="axes fraction", size=10)
            axx[0, i].annotate(
                labels[i].format(t), xy=(0.05, 0.05), xycoords="axes fraction", size=10
            )
            axx[0, i].scatter(xt[t, :, 0], xt[t, :, 1], s=size, c=colour)
            axx[0, i].plot(rmax * cos(th), rmax * sin(th), lw=1, color="red", alpha=0.2)
        else:  # bottom row
            axx[1, i - 3].set_xlim(-2.5 * r_mean, 2.5 * r_mean)
            axx[1, i - 3].set_ylim(-2.5 * r_mean, 2.5 * r_mean)
            axx[1, i - 3].set_aspect("equal")
            axx[1, i - 3].annotate(avg_dist[i], xy=(0.05, 0.93), xycoords="axes fraction", size=10)
            axx[1, i - 3].annotate(
                labels[i].format(t), xy=(0.05, 0.05), xycoords="axes fraction", size=10
            )
            axx[1, i - 3].scatter(xt[t, :, 0], xt[t, :, 1], s=size, c=colour)
            axx[1, i - 3].plot(
                rmax * cos(th), rmax * sin(th), lw=1, color="red", alpha=0.2
            )
    figx.tight_layout()
    if save:
        figx.savefig(name.format("xy", "png"))


# X-Z
if xz_plot:
    figz, axz = plt.subplots(2, 3, sharex=True, sharey=True)
    figz.suptitle("x-z axes")
    for i in range(6):
        t = int(tlabels[i])
        if i < 3:  # top row
            axz[0, i].set_aspect("equal")
            axz[0, i].set_title(avg_dist[i], size=10)
            axz[0, i].annotate(
                labels[i].format(t), xy=(0.05, 0.05), xycoords="axes fraction", size=10
            )
            axz[0, i].scatter(xt[t, :, 0], xt[t, :, 1], s=size, c=colour)
            axz[0, i].plot(rmax * cos(th), rmax * sin(th), lw=1, color="red", alpha=0.2)
        else:  # bottom row
            axz[1, i - 3].set_aspect("equal")
            axz[1, i - 3].set_title(avg_dist[i], size=10)
            axz[1, i - 3].annotate(
                labels[i].format(t), xy=(0.05, 0.05), xycoords="axes fraction", size=10
            )
            axz[1, i - 3].scatter(xt[t, :, 0], xt[t, :, 1], s=size, c=colour)
            axz[1, i - 3].plot(
                rmax * cos(th), rmax * sin(th), lw=1, color="red", alpha=0.2
            )
    figz.tight_layout()
    if save:
        figz.savefig(name.format("xy", "png"))
if save:
    print("saved")
