import numpy as np
import gravsim as gs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
rand dist of theta, phi:
r1 = random(0,1)
r2 = random(0,1)
theta = 2pi * r1
phi = cos^-1(2*r2 - 1)
'''

#seeding so each time it runs we get the same values
np.random.seed(1998)

# initial values/arrays
nsteps = 10000
timestep = 0.0001
softening = 2
n_halo = 125
n_disc = 124
m_centre = np.full((1),25)
m_disc = np.ones((n_disc))
m_halo = np.full((n_halo),1.5)
ms = np.concatenate((m_centre,m_disc,m_halo))

# centre coords
centre = np.zeros((1,3))

# disc radii are easy to generate because density function is simple
random_disc = np.random.uniform(0.001,1, size = n_disc)
r_disc = -3. * np.log(random_disc)

# halo radii are hard to generate because density function is nasty
x = np.linspace(0.001,30, num = 20000)

def p(x): # pdf
    return (x**2) * (1/20) * ((x** (-2)) - (1/900))

def q(x): # nicer pdf (of uniform dist) with q(x) > p(x) for all x
    return 0.05

dqp = q(x) - p(x)

if any(x < 0 for x in dqp) == True:
    print('YOU FUCKED UP, CHECK HALO q(x) FUNCTION')
k = max(p(x) / q(x)) # multiplication factor to make q(x) close to p(x)

def rejection_sampling(iter=1000):
    samples = []
    for i in range(iter):
        z = np.random.uniform(0.0001,30)
        u = np.random.uniform(0, k*q(z))
        if u <= p(z):
            samples.append(z)
    return np.array(samples)

r_halo_options = rejection_sampling(iter = 100000)
r_halo = np.empty((n_halo))

for i in range(n_halo): # pick random values from the distribution
    r_halo[i] = np.random.choice(r_halo_options, replace = False)

# points
    # disc
disc_thetas = np.random.uniform(0, 2*np.pi, size = n_disc)
disc_pts = np.zeros((n_disc,3))
disc_pts[:,0] = r_disc * np.cos(disc_thetas)
disc_pts[:,1] = r_disc * np.sin(disc_thetas)

    # halo
halo_theta = np.random.uniform(0,2*np.pi, size = n_halo)
halo_phi = np.arccos(2 * np.random.random(size = n_halo) - 1)
halo_pts = np.empty((n_halo,3))
halo_pts[:,0] = r_halo * np.cos(halo_theta) * np.sin(halo_phi)
halo_pts[:,1] = r_halo * np.sin(halo_theta) * np.sin(halo_phi)
halo_pts[:,2] = r_halo * np.cos(halo_phi)

    # combining
points = np.empty((n_halo + n_disc + 1, 3))
points[0,:] = centre
points[1:n_disc + 1,:] = disc_pts
points[n_disc + 1:,:] = halo_pts

# velocities
forces_initial, gpe_initial = gs.force_calculator(ms, points, softening)
velocities = np.zeros((1 + n_disc + n_halo, 3))
    # disc
f_disc_init = forces_initial[1:n_disc + 1]
v_disc_init = np.empty((n_disc)) # circular velocity magnitude
u_disc_init = np.empty((n_disc)) # isotropic velocity magnitude
velocity_disc = np.zeros((n_disc,3))

for i in range(n_disc):
    
    #circular
    force_mean = np.mean(f_disc_init[i,0:2])
    v_disc_init[i] = (r_disc[i] * np.absolute(force_mean)) ** 0.5
    
    #isotropic
    r_close = r_disc[i] - 2
    if r_close <= 0:
        r_close = 0.
    r_far = r_disc[i] + 2
    dr = r_far - r_close
    far_pos = np.array([r_far * np.cos(disc_thetas[i]),
                          r_far * np.sin(disc_thetas[i]),
                          0])
    close_pos = np.array([r_close * np.cos(disc_thetas[i]),
                          r_close * np.sin(disc_thetas[i]),
                          0])
    far_force = np.mean(gs.force_point(far_pos, ms, points, softening)[0])
    close_force = np.mean(gs.force_point(close_pos,ms, points, softening)[0])
    df_dr = (far_force - close_force) / dr
    kappa = (3 * np.absolute(force_mean) / r_disc[i]) + df_dr
    u_disc_init[i] = 1.344 * np.exp(-1 * r_disc[i] / 3) / kappa
    
#circular velocity
velocity_disc[:,0] += -1 * v_disc_init * np.sin(disc_thetas)
velocity_disc[:,1] += v_disc_init * np.cos(disc_thetas)
    
#isotropic velocity
u_theta = np.random.uniform(0,2*np.pi, size = n_disc)
u_phi = np.arccos(2 * np.random.random(size = n_disc) - 1)
velocity_disc[:,0] += u_disc_init * np.cos(u_theta) * np.sin(u_phi)
velocity_disc[:,1] += u_disc_init * np.sin(u_theta) * np.sin(u_phi)
velocity_disc[:,2] += u_disc_init * np.cos(u_phi)

velocities[1:n_disc + 1,:] = velocity_disc

    # halo
velocity_halo = np.zeros((n_halo,3))
v_mag_halo = (0.5 * np.absolute(gpe_initial[n_disc + 1:])) ** 0.5
v_halo_theta = np.random.uniform(0,2*np.pi)
v_halo_phi = np.arccos(2 * np.random.random() - 1)
velocity_halo[:,0] += v_mag_halo * np.cos(v_halo_theta) * np.sin(v_halo_phi)
velocity_halo[:,1] += v_mag_halo * np.sin(v_halo_theta) * np.sin(v_halo_phi)
velocity_halo[:,2] += v_mag_halo * np.cos(v_halo_phi)

velocities[n_disc + 1:,:] = velocity_halo

r_times, v_times, f_times, kinetic, gpe = gs.timeloop(nsteps,
                                                      timestep,
                                                      ms,
                                                      points,
                                                      velocities,
                                                      softening)

# trajectory plots
fig_tr = plt.figure()
ax_tr = fig_tr.add_subplot(111)

for i in range(1, 1 + n_disc):
    ax_tr.scatter(r_times[:,i,0], r_times[:,i,1],
                s = 2, ls = '-', lw = 1)
ax_tr.scatter(r_times[0,1:n_disc + 1,0],r_times[0,1:n_disc + 1,1],
              s = 8, marker = '^', c = 'black')
ax_tr.scatter(r_times[-1,1:n_disc + 1,0],r_times[-1,1:n_disc + 1,1],
              s = 8, marker = '^', c = 'red')
fig_tr.show()

if __name__ == '__main__':
    
    # initial position plots
    '''
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    
    ax.scatter(0,0,0, s = 25, # centre particle
               c = 'black')
    ax.scatter(halo_pts[:,0], halo_pts[:,1], halo_pts[:,2], # halo
               s = 10, c = 'red')
    ax.scatter(disc_pts[:,0], disc_pts[:,1], disc_pts[:,2], # disc
               s = 5, c = 'blue')
    
    fig2, ax2 = plt.subplots(1,2)
    ax2[0].hist(points[:,0],bins = 400, density = True, color = 'blue')
    ax2[1].hist(points[:,1], bins = 400, density = True, color = 'red')'''