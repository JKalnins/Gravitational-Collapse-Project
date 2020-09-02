from poisson_creator import poisson_distributor
import numpy as np
import matplotlib.pyplot as plt

ns = np.arange(100,1100,50)
rmax = 1e18
ndims = 3
rmeans = []
for i, n in enumerate(ns):
    rmin = ((rmax ** 3) / n) ** (1 / 3)
    rep_mean = []
    for rep in range(10):
        pts = poisson_distributor(n, rmax, rmin, ndims, report = False)
        rs = np.linalg.norm(pts, axis = 1)
        rep_mean.append(np.mean(rs))
    rmeans.append(np.mean(rep_mean))

plt.scatter(ns, rmeans)