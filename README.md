# Gravitational-Collapse-Project
3rd Year University Computing Project Module
Direct N-Body simulation of gravitational attraction using Python

## Workflow
poisson_creator.py: generates a random distribution of particles in 1-3D using a Poisson-sphere distribution to avoid particles beginning extremely close to one another

gravsim.py: direct physics simulation of gravitational force between N bodies using a Leapfrog integration method which provides 2nd-order accuracy with no extra calculations compared to an Euler method. Numba's Just-In-Time compilation using the @jit(nopython=True) wrapper allows an increase in speed of 10-100x on Python computations, even with numpy arrays

poisson_sim_and_save.py: Creates a set of particles using poisson_creator then uses gravsim to actually simulate the system over time, then saves the information using numpy's .npz format to allow for easy retrieval of arrays for analysis. Separating analysis from computation means data doesn't have to be repeatedly collected, and allows multiple runs to be aggregated for comparison (at the expense of some storage space)
