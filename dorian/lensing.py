import concurrent.futures
import multiprocessing
from multiprocessing import shared_memory

import numpy as np
import healpy as hp
from .cosmology import d_c
from .raytracing import raytrace
from .logging import info, success, warning
from math import prod

def _raytrace_worker(args: tuple) -> dict:
    """Worker for parallel ray-tracing of a single source redshift.

    Receives a lightweight tuple ``(z_s, shm_name, shm_shape, shm_dtype,
    distances, redshifts_clean, omega_m, omega_l, nside, interp, lmax,
    parallel_transport, nufft_nthreads)``.

    The large ``shells`` array is read from a named shared-memory block
    (zero-copy; no pickling of the pixel data).
    """
    (z_s, shm_name, shm_shape, shm_dtype,
     distances, redshifts_clean,
     omega_m, omega_l, nside, interp, lmax,
     parallel_transport, nufft_nthreads) = args

    import dorian.logging as _dlog
    _dlog._prefix = f"[z_s={z_s:.2f}] "

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    try:
        shells_2d = np.ndarray(shm_shape, dtype=shm_dtype, buffer=existing_shm.buf)
        shells = list(shells_2d)   # list of 1-D views — exactly what raytrace() expects
        kappa_born, A_final, beta_final, theta = raytrace(
            shells=shells,
            z_s=z_s,
            omega_m=omega_m,
            omega_l=omega_l,
            nside=nside,
            shell_redshifts=redshifts_clean,
            shell_distances=distances,
            interp=interp,
            lmax=lmax,
            parallel_transport=parallel_transport,
            nthreads=nufft_nthreads,
        )
    finally:
        existing_shm.close()

    kappa_raytraced = 1.0 - 0.5 * (A_final[0, 0] + A_final[1, 1])
    n_used = sum(1 for z in redshifts_clean if z < z_s)
    return {
        'convergence_born': kappa_born,
        'convergence_raytraced': kappa_raytraced,
        'distortion_matrix': A_final,
        'ray_positions': beta_final,
        'initial_positions': theta,
        'shell_info': {
            'redshifts': redshifts_clean,
            'distances': distances,
            'n_shells_total': len(redshifts_clean),
            'n_shells_used': n_used,
        },
    }


def prepare_density_shells(
    density_maps,
    redshifts,
    box_size,
    n_particles,
    omega_m,
    h=0.6766,
    omega_l=None,
    nside=None,
    shell_widths=None,
):
    """
    Convert density planes to Dorian mass format for ray-tracing.

    This function handles the conversion from various density representations
    (particle number density, overdensity, or mass) to the mass per pixel format
    expected by Dorian's ray-tracing algorithm.

    Parameters
    ----------
    density_maps : list of array-like
        List of HEALPix density maps for each shell. Each map should be a 1D
        array with ``npix = 12 * nside**2`` elements in RING ordering.
    redshifts : array-like
        Redshift of each shell, must have same length as ``density_maps``.
    box_size : float or tuple of float
        Simulation box size in Mpc/h. If a single value, assumes cubic box.
        If tuple, should be ``(Lx, Ly, Lz)``.
    n_particles : int
        Total number of particles in the simulation (e.g., ``512**3``).
    omega_m : float
        Matter density parameter (Omega_m).
    h : float, optional
        Dimensionless Hubble parameter H0/(100 km/s/Mpc). Default: 0.6766.
    omega_l : float, optional
        Dark energy density parameter (Omega_Lambda). Default: ``1 - omega_m``.
    nside : int, optional
        Target HEALPix NSIDE resolution. If None, auto-detected from first map.
    shell_widths : array-like, optional
        Shell thickness d_R per shell in Mpc/h. Required when density_maps
        contain number density (particles per (Mpc/h)^3). When provided, the
        pixel volume is computed per shell to convert density to mass. When
        None, density_maps are assumed to already contain particle counts per
        pixel and are directly multiplied by the particle mass.

    Returns
    -------
    shells : list of np.ndarray
        Mass per pixel in units of 10^10 M_sun/h, ready for ray-tracing.
    shell_distances : list of float
        Comoving distances to each shell in Mpc/h.
    shell_redshifts : list of float
        Validated redshifts for each shell.

    Examples
    --------
    Basic usage with number density maps and shell widths:

    >>> import numpy as np
    >>> from dorian.lensing import prepare_density_shells
    >>> # Create mock density maps (4 shells, nside=64)
    >>> nside = 64
    >>> npix = 12 * nside**2
    >>> density_maps = [np.random.randn(npix) * 0.01 for _ in range(4)]
    >>> redshifts = [0.1, 0.2, 0.3, 0.4]
    >>> shell_widths = [100.0, 100.0, 100.0, 100.0]  # Mpc/h
    >>> shells, distances, z_out = prepare_density_shells(
    ...     density_maps=density_maps,
    ...     redshifts=redshifts,
    ...     box_size=1000.0,        # Mpc/h
    ...     n_particles=256**3,
    ...     omega_m=0.3,
    ...     shell_widths=shell_widths,
    ... )
    >>> len(shells)
    4

    See Also
    --------
    raytrace_from_density : Combines preparation and ray-tracing in one call.
    dorian.raytracing.raytrace : Low-level ray-tracing function.
    """
    if omega_l is None:
        omega_l = 1.0 - omega_m

    assert shell_widths is not None, "shell_widths must be provided when density_maps contain number density"

    if isinstance(box_size, (int, float)):
        box_size = (box_size, box_size, box_size)

    if nside is None:
        nside = hp.npix2nside(len(density_maps[0]))

    npix = hp.nside2npix(nside)

    rho_crit_h2 = 2.775e11
    rho_crit = rho_crit_h2 * h**2
    rho_matter = omega_m * rho_crit
    volume_box_mpc = prod(box_size) / h**3
    particle_mass_msun = (rho_matter * volume_box_mpc) / n_particles
    particle_mass_dorian = (particle_mass_msun * h) / 1e10

    info(f"Preparing {len(density_maps)} density shells: nside={nside}, "
         f"particle mass={particle_mass_dorian:.4e} [10^10 M_sun/h]")

    shells = []
    shell_distances = []
    shell_redshifts = []

    for i, (density_map, z) in enumerate(zip(density_maps, redshifts)):
        map_nside = hp.npix2nside(len(density_map))
        if map_nside != nside:
            warning(f"Shell {i}: ud_grading nside {map_nside} -> {nside}")
            density_map = hp.ud_grade(density_map, nside, power=-2)

        d_k = d_c(z=z, Omega_M=omega_m, Omega_L=omega_l)
        info(f"  Shell {i+1}/{len(density_maps)}: z={z:.4f}, d={d_k:.1f} Mpc/h")

        dr = shell_widths[i]
        R_min = max(d_k - dr / 2, 0.0)
        R_max = d_k + dr / 2
        shell_vol = (4.0 / 3.0) * np.pi * (R_max**3 - R_min**3)
        pixel_vol = shell_vol / npix
        mass_per_pixel = density_map * pixel_vol * particle_mass_dorian
  
        shells.append(np.asarray(mass_per_pixel, dtype=np.float64))
        shell_distances.append(float(d_k))
        shell_redshifts.append(float(z))

    success(f"Prepared {len(shells)} density shells")

    return shells, shell_distances, shell_redshifts


def raytrace_from_density(
    density_maps,
    redshifts,
    z_source,
    box_size,
    n_particles,
    omega_m,
    h=0.6766,
    omega_l=None,
    nside=None,
    interp='bilinear',
    shell_widths=None,
    parallel_transport=True,
    lmax=0,
    nufft_nthreads=1,
    n_workers=None,
):
    """
    Perform full-sky weak lensing ray-tracing from density maps.

    This is the main high-level interface for Dorian. It combines density shell
    preparation and ray-tracing into a single call, making it easy to compute
    weak lensing convergence maps from simulation lightcones.

    Parameters
    ----------
    density_maps : list of array-like
        List of HEALPix density maps for each shell. Each map should be a 1D
        array with ``npix = 12 * nside**2`` elements in RING ordering.
    redshifts : array-like
        Redshift of each shell, must have same length as ``density_maps``.
        Only shells with ``z < z_source`` will be used.
    z_source : float or list of float
        Source redshift(s). When a scalar, traces rays for a single source and
        returns a single dict. When a list, calls ``prepare_density_shells``
        exactly once and runs each source in parallel via
        ``ProcessPoolExecutor``, returning a merged dict with stacked arrays of
        shape ``(n_sources, ...)``.
    box_size : float or tuple of float
        Simulation box size in Mpc/h. If a single value, assumes cubic box.
    n_particles : int
        Total number of particles in the simulation (e.g., ``512**3``).
    omega_m : float
        Matter density parameter (Omega_m).
    h : float, optional
        Dimensionless Hubble parameter H0/(100 km/s/Mpc). Default: 0.6766.
    omega_l : float, optional
        Dark energy density parameter (Omega_Lambda). Default: ``1 - omega_m``.
    nside : int, optional
        Target HEALPix NSIDE resolution. If None, auto-detected from first map.
    interp : {'bilinear', 'ngp', 'nufft'}, optional
        Interpolation method for sampling deflection fields at ray positions:

        - ``'bilinear'``: Bilinear interpolation (default, good balance)
        - ``'ngp'``: Nearest grid point (fastest, lowest accuracy)
        - ``'nufft'``: Non-uniform FFT (highest accuracy, slowest)

    shell_widths : array-like, optional
        Shell thickness d_R per shell in Mpc/h. Required when density_maps
        contain number density (particles per (Mpc/h)^3). Passed through to
        ``prepare_density_shells``.
    parallel_transport : bool, optional
        Whether to apply parallel transport of the distortion matrix along
        geodesics. Recommended to keep True for accurate results. Default: True.
    lmax : int, optional
        Maximum multipole ell for spherical harmonic transforms.
        Default: ``3 * nside`` (sufficient for most applications).
    nufft_nthreads : int, optional
        Number of OpenMP threads for ``'nufft'`` interpolation. Default: 1.
    n_workers : int or None, optional
        Number of worker processes for parallel multi-source execution.
        Only used when ``z_source`` is a list. Defaults to ``len(z_source)``
        (one worker per source).

    Returns
    -------
    results : dict
        Dictionary containing all ray-tracing outputs.

        **Scalar z_source** (existing behaviour):

        - ``'convergence_born'`` : np.ndarray, shape ``(npix,)``
        - ``'convergence_raytraced'`` : np.ndarray, shape ``(npix,)``
        - ``'distortion_matrix'`` : np.ndarray, shape ``(2, 2, npix)``
        - ``'ray_positions'`` : np.ndarray, shape ``(2, npix)``
        - ``'initial_positions'`` : np.ndarray, shape ``(2, npix)``
        - ``'shell_info'`` : dict with keys ``'redshifts'``, ``'distances'``,
          ``'n_shells_total'``, ``'n_shells_used'``.

        **List z_source** (multi-source parallel path):

        - ``'convergence_born'`` : np.ndarray, shape ``(n_sources, npix)``
        - ``'convergence_raytraced'`` : np.ndarray, shape ``(n_sources, npix)``
        - ``'distortion_matrix'`` : np.ndarray, shape ``(n_sources, 2, 2, npix)``
        - ``'ray_positions'`` : np.ndarray, shape ``(n_sources, 2, npix)``
        - ``'initial_positions'`` : np.ndarray, shape ``(n_sources, 2, npix)``
        - ``'shell_info'`` : list of dict, one per source.

    Examples
    --------
    Basic ray-tracing from a simulation lightcone:

    >>> import numpy as np
    >>> from dorian.lensing import raytrace_from_density
    >>> # Load your density maps (list of HEALPix maps)
    >>> density_maps = [...]  # 4 shells at different redshifts
    >>> redshifts = [0.1, 0.2, 0.3, 0.4]
    >>> results = raytrace_from_density(
    ...     density_maps=density_maps,
    ...     redshifts=redshifts,
    ...     z_source=1.0,
    ...     box_size=2000.0,       # Mpc/h
    ...     n_particles=512**3,
    ...     omega_m=0.3,
    ...     nside=512,
    ... )
    >>> kappa_born = results['convergence_born']
    >>> kappa_raytraced = results['convergence_raytraced']

    Extract shear from the distortion matrix:

    >>> A = results['distortion_matrix']
    >>> gamma1 = 0.5 * (A[0, 0] - A[1, 1])  # shear component 1
    >>> gamma2 = A[0, 1]                     # shear component 2

    Using high-precision NUFFT interpolation:

    >>> results = raytrace_from_density(
    ...     density_maps=density_maps,
    ...     redshifts=redshifts,
    ...     z_source=1.0,
    ...     box_size=2000.0,
    ...     n_particles=512**3,
    ...     omega_m=0.3,
    ...     interp='nufft',
    ...     nufft_nthreads=8,
    ... )

    Multi-source parallel ray-tracing (outputs have an extra leading axis):

    >>> z_sources = [0.5, 1.0, 2.0]
    >>> results = raytrace_from_density(
    ...     density_maps=density_maps,
    ...     redshifts=redshifts,
    ...     z_source=z_sources,   # list → parallel execution
    ...     box_size=2000.0,
    ...     n_particles=512**3,
    ...     omega_m=0.3,
    ...     nside=512,
    ...     n_workers=3,          # one process per source
    ... )
    >>> results['convergence_born'].shape   # (3, npix)
    (3, 3145728)
    >>> results['distortion_matrix'].shape  # (3, 2, 2, npix)
    (3, 2, 2, 3145728)

    Notes
    -----
    The convergence is computed two ways:

    1. **Born approximation** (``convergence_born``): Integrates kappa along
       unperturbed (straight) ray paths. Fast but ignores lens-lens coupling.

    2. **Ray-traced** (``convergence_raytraced``): Computed from the full
       distortion matrix as ``kappa = 1 - 0.5 * Tr(A)``. Includes all
       post-Born corrections.

    See Also
    --------
    prepare_density_shells : Convert density maps to mass format separately.
    dorian.raytracing.raytrace : Low-level ray-tracing function.
    """
    if omega_l is None:
        omega_l = 1.0 - omega_m

    # Detect multi-source mode using robust scalar and sequence checks.
    # - Scalars: np.isscalar(...) or 0-D numpy arrays -> single source.
    # - Multi-source: explicit sequences (list/tuple) or numpy arrays with ndim > 0.
    if np.isscalar(z_source) or (isinstance(z_source, np.ndarray) and z_source.ndim == 0):
        is_multi = False
    elif isinstance(z_source, (list, tuple, np.ndarray)):
        # For numpy arrays, ndim > 0 is guaranteed here since 0-D case is handled above.
        is_multi = True
    else:
        # Fallback: treat unknown types as single source to avoid misclassification.
        is_multi = False

    info(f"raytrace_from_density: z_source={z_source}, interp='{interp}', "
         f"nside={nside if nside else 'auto'}, multi={is_multi}")

    shells, distances, redshifts_clean = prepare_density_shells(
        density_maps=density_maps,
        redshifts=redshifts,
        box_size=box_size,
        n_particles=n_particles,
        omega_m=omega_m,
        h=h,
        omega_l=omega_l,
        nside=nside,
        shell_widths=shell_widths,
    )

    if nside is None:
        nside = hp.npix2nside(len(shells[0]))

    if is_multi:
        z_sources = list(z_source)
        if n_workers is None:
            n_workers = min(len(z_sources), multiprocessing.cpu_count())

        # Stack shells into one contiguous 2-D array for shared memory.
        shells_2d = np.array(shells)          # shape: (n_shells, npix), float64
        shm = shared_memory.SharedMemory(create=True, size=shells_2d.nbytes)
        try:
            # Copy into the shared block (write once in parent).
            shm_array = np.ndarray(shells_2d.shape, dtype=shells_2d.dtype, buffer=shm.buf)
            shm_array[:] = shells_2d

            worker_args = [
                (float(z_s), shm.name, shells_2d.shape, shells_2d.dtype.str,
                 distances, redshifts_clean,
                 omega_m, omega_l, nside, interp, lmax, parallel_transport, nufft_nthreads)
                for z_s in z_sources
            ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(_raytrace_worker, worker_args))
        finally:
            shm.close()
            shm.unlink()

        success(f"raytrace_from_density complete: {len(z_sources)} sources processed in parallel")

        return {
            'convergence_born':      np.stack([r['convergence_born']      for r in results]),  # (n_sources, npix)
            'convergence_raytraced': np.stack([r['convergence_raytraced'] for r in results]),  # (n_sources, npix)
            'distortion_matrix':     np.stack([r['distortion_matrix']     for r in results]),  # (n_sources, 2, 2, npix)
            'ray_positions':         np.stack([r['ray_positions']         for r in results]),  # (n_sources, 2, npix)
            'initial_positions':     np.stack([r['initial_positions']     for r in results]),  # (n_sources, 2, npix)
            'shell_info':            [r['shell_info'] for r in results],                       # list[dict]
        }
    else:
        kappa_born, A_final, beta_final, theta = raytrace(
            shells=shells,
            z_s=z_source,
            omega_m=omega_m,
            omega_l=omega_l,
            nside=nside,
            shell_redshifts=redshifts_clean,
            shell_distances=distances,
            interp=interp,
            lmax=lmax,
            parallel_transport=parallel_transport,
            nthreads=nufft_nthreads,
        )

        kappa_raytraced = 1.0 - 0.5 * (A_final[0, 0] + A_final[1, 1])

        n_used = sum(1 for z in redshifts_clean if z < z_source)

        success(f"raytrace_from_density complete: {n_used}/{len(redshifts_clean)} shells used")

        return {
            'convergence_born': kappa_born,
            'convergence_raytraced': kappa_raytraced,
            'distortion_matrix': A_final,
            'ray_positions': beta_final,
            'initial_positions': theta,
            'shell_info': {
                'redshifts': redshifts_clean,
                'distances': distances,
                'n_shells_total': len(redshifts_clean),
                'n_shells_used': n_used,
            },
        }
