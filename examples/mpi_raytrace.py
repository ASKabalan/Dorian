#!/usr/bin/env python
"""
MPI ray-tracing example. Run with:

    mpirun -n 4 python mpi_raytrace.py

Each MPI rank processes a subset of z_sources (round-robin).
Results are gathered to rank 0 and saved to results.npz.
"""
from mpi4py import MPI
import numpy as np
import os
from dorian.lensing import raytrace_from_density

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ── Load data (all ranks load independently) ──────────────────────────────────
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data")
data = np.load(os.path.join(SAMPLE_DIR, "data.npz"))
lc          = data['lc']
redshifts   = data['redshift']
width       = data['width']
box_size    = tuple(data['box_size'])
n_particles = int(data['n_particles'])

h       = 0.6774
omega_m = 0.0486 + 0.2589   # Omega_b + Omega_c
omega_l = 1.0 - omega_m

z_sources = np.linspace(0.1 , 1.5 , 32)  # 32 source planes from z=0.1 to z=1.5
nside     = 1024

if rank == 0:
    print(f"MPI size={size}, {len(z_sources)} z_sources, nside={nside}")

# ── Ray-trace ─────────────────────────────────────────────────────────────────
results = raytrace_from_density(
    density_maps=lc,
    redshifts=redshifts,
    z_source=z_sources,
    box_size=box_size,
    shell_widths=width,
    n_particles=n_particles,
    omega_m=omega_m,
    h=h,
    omega_l=omega_l,
    nside=nside,
    interp='bilinear',
    comm=comm,
)

# ── Save results (rank 0 only) ────────────────────────────────────────────────
if rank == 0:
    np.savez(
        "results.npz",
        convergence_born=results['convergence_born'],
        convergence_raytraced=results['convergence_raytraced'],
        z_sources=z_sources,
    )
    print("Saved results to results.npz")
    for i, z in enumerate(z_sources):
        kb = results['convergence_born'][i]
        kr = results['convergence_raytraced'][i]
        print(f"  z={z:.1f}: kappa_born mean={kb.mean():.4e}, kappa_rt mean={kr.mean():.4e}")
