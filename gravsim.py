import numpy as np
from numba import jit

"""
units:
    G = 1
All other units can be defined as needed
"""
G = 6.67430e-11
# G = 1


@jit(nopython=True)
def force_point(location, m, masses, positions, eps, soft):
    """
    calculates gravitational force & GPE on an object at a location
    * both are softened by epsilon *
    Inputs:
        location: 1x3 array (location of point you want to measure, cartesian)
        m: float (mass of object)
        masses: Nx1 array
        positions: Nx3 array (cartesian)
        eps: float (softening constant) [assumes eps = 0]
        soft: bool (True = softening considered, False = not)
    Outputs:
        force: 1x3 array
        gpe: float
    """
    force = np.zeros((1, 3))
    gpe = 0.0
    for j in range(len(masses)):
        rj = positions[j, :]
        dr = rj - location  # vector distance between location & j
        dist_sq = (dr ** 2).sum()  # scalar distance between i & j squared
        softened_dist_sq = dist_sq + (eps ** 2)
        force_j = dr * (G * m * masses[j] / (softened_dist_sq ** (3 / 2)))
        force += force_j
        if soft:
            gpe -= G * m * masses[j] / np.sqrt(softened_dist_sq)
        else:
            gpe -= G * m * masses[j] / np.sqrt(dist_sq)
    return (force, gpe)


@jit(nopython=True)
def find_com(positions, masses):
    """
    finds com of a set of particles in 3D
    Inputs:
        pos: nx3 array
        masses: nx1 array
    Outputs:
        com: 1x3 array
    """
    com = np.zeros((3))
    m_sum = np.sum(masses)
    for i, m in enumerate(masses):
        com += m * positions[i,:]
    com = com / m_sum
    return(com)


@jit(nopython=True)
def force_calculator(masses, positions, eps):

    """
    calculates: force due to gravity between N objects.
    Inputs:
        masses: Nx1 array
        positions: Nx3 array (cartesian)
        velocities: Nx3 array (only used for working out velocity
        of combined particles)
        combined: dict - set of combined particles
        eps: float (softening constant) [assumes eps = 0]
    Outputs:
        forces: Nx3 array (cartesian)
        gpe: Nx1 array
    """
    forces = np.zeros_like(positions)  # 3-vector
    gpe = 0  # scalar

    for i in range(len(masses)):
        for j in range(i + 1, len(masses)):
            ri = positions[i, :]
            rj = positions[j, :]
            dr = rj - ri  # vector distance between i & j
            dist_sq = np.sum(dr ** 2)  # scalar distance between i & j squared
            softened_dist = (dist_sq + (eps ** 2)) ** (3 / 2)
            force_ij = dr * (G * masses[i] * masses[j] / softened_dist)
            forces[i, :] += force_ij
            forces[j, :] += force_ij * -1
            gpe += -1 * G * masses[i] * masses[j] / np.sqrt(dist_sq + eps ** 2)
    return (forces, gpe)


@jit(nopython=True)
def one_step(timestep, masses, mass3, positions, velocities, eps):
    """
    takes a set of masses, with positions & velocities, calc's force on them
    then finds positions & velocities at next timestep, using leapfrog
    integration method
    Inputs:
        timestep: float
        masses: Nx1 array
        mass3: Nx3 array (for matrix ops see below)
        positions: Nx3 array (cartesian)
        velocities: Nx3 array (cartesian)
        eps: float (softening constant) [assume eps = 0]
    currently, can't figure out how to add an ext field as an optional variable
    with numba (type is specified at runtime, so can't fuck with it)
    Outputs:
        r_update: Nx3 array of new positions (cartesian)
        v_update: Nx3 array of new velocities (cartesian)
        forces: Nx3 array of new forces (cartesian)
        kinetic: Nx1 array of KE's for position 2
        gpe: Nx1 array of GPE's for position 1
        new_com: 1x3 array of current COMs
        r_com_frame: Nx3 array of new positions with com at (0,0,0)
    mass3 is an Nx3 array of masses ie. {[m1 m1 m1] / [m2 m2 m2] / ...}
    this allows matrix ops to be used to update velocity & position which are
    quicker than looping
    """
    forces, gpe = force_calculator(masses, positions, eps)
    v_update = velocities + (forces * timestep / mass3)
    v_half = velocities + (forces * 0.5 * timestep / mass3)
    v_half_mag_sq = np.array([(v_half[i, :] ** 2).sum() for i in range(len(masses))])
    r_update = positions + v_update * timestep
    kinetic = 0.5 * masses * v_half_mag_sq
    new_com = find_com(positions, masses)
    r_com_frame = np.empty_like(positions)
    for i in range(len(masses)):
        r_com_frame[i, :] = positions[i] - new_com
    return (r_update, v_update, forces, kinetic, gpe, new_com, r_com_frame)


@jit(nopython=True)
def timeloop(nsteps, timestep, masses, positions, velocities, eps):
    """
    repeats the one-step process for n steps and records params at each step
    into Nx3x(nsteps) arrays
    Inputs:
        nsteps: int
        timestep: float
        masses: Nx1 array
        positions: Nx3 array (cartesian)
        velocities: Nx3 array (cartesian)
        eps: float (softening constant) [assume eps = 0]
    Outputs:
        r_times: Nx3x(nsteps) array (positions)
        v_times: Nx3x(nsteps) array (velocities)
        f_times: Nx3x(nsteps) array (forces)
        kinetic: Nx1x(nsteps) array
        gpe: Nx1x(nsteps) array
        com_time: (nsteps)x3 array (coms)
        r_com_frame_t: Nx3x(nsteps) array (positions rel to com)
    """
    kinetic = np.empty((nsteps + 1, len(masses)))
    com_time = np.zeros((nsteps + 1, 3))
    com_time[0,:] = find_com(positions, masses)
    kinetic[0, :] = (
        0.5
        * masses
        * np.array([(velocities[i, :] ** 2).sum() for i in range(len(masses))])
    )
    gpe = np.empty((nsteps + 1))
    """
    in one_step, GPE is calc'ed for starting position but KE is calc'ed
    after the advance; so ke[0] must be calc'ed here. We will not have a final
    GPE until it's specifically calculated at the end.
    """
    r_times = np.zeros((nsteps + 1, len(masses), 3))
    r_com_frame_t = np.zeros((nsteps + 1, len(masses), 3))
    v_times = np.zeros((nsteps + 1, len(masses), 3))
    f_times = np.zeros((nsteps + 1, len(masses), 3))

    r_times[0, :, :] += positions
    r_com_frame_t[0, :, :] += positions
    v_times[0, :, :] += velocities
    if v_times[0,0,0] != 0:
        print("movement on the black!")
    """
    this creates Nx3x(nsteps) arrays by making a list of arrays then using
    np.stack to form an array from the list, after setting the 1st array to
    the initial input values
    Reference r/v_times with [timestep, particle no, axis no]
    e.g. 1st timestep, 5th particle, y axis: [0,4,1]
    """
    mass3 = np.zeros((len(masses), 3))
    for i, mass in enumerate(masses):
        mass3[i, :] = [mass for p in range(3)]

    for n in range(nsteps):
        (
            r_times[n + 1, :, :], # we use n+1 for these because they update
            v_times[n + 1, :, :],
            f_times[n, :, :], # we use n for these because they're pre-update
            kinetic[n + 1, :],
            gpe[n],
            com_time[n + 1, :],
            r_com_frame_t[n + 1, :, :]
        ) = one_step(
            timestep, masses, mass3, r_times[n, :, :], v_times[n, :, :], eps
        )
    f_times[-1, :, :], gpe[-1] = force_calculator(masses, r_times[-1, :, :], eps)
    # for final step, we need to find the pre-update var's (i.e. F & gpe)
    return (r_times, v_times, f_times, kinetic, gpe, com_time, r_com_frame_t)


if __name__ == "__main__":
    objs = 10
    dt = 100
    steps = 3
    rad = 1e16  # approx width of norm dist for x
    vel = 0  # approx width of norm dist for v
    eps = 10
    m = np.random.normal(1.0, 0.14, objs)
    x = np.random.normal(0, size=(objs, 3)) * rad
    v = np.random.normal(0, size=(objs, 3)) * vel
    x2, v2, f, k, g, coms = timeloop(steps, dt, m, x, v, eps)
    for com in coms:
        print(com)