import numpy as np
import healpy as hp
from .cosmology import d_c
from .raytracing import raytrace


def prepare_density_shells(
    density_maps,
    redshifts,
    box_size,
    n_particles,
    omega_m,
    h=0.6766,
    omega_l=None,
    nside=None,
    unit='number_density',
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
    unit : {'number_density', 'overdensity', 'mass'}, optional
        Input unit type:

        - ``'number_density'``: Particle count per pixel (default)
        - ``'overdensity'``: delta = rho/rho_mean - 1
        - ``'mass'``: Already in 10^10 M_sun/h per pixel

    Returns
    -------
    shells : list of np.ndarray
        Mass per pixel in units of 10^10 M_sun/h, ready for ray-tracing.
    shell_distances : list of float
        Comoving distances to each shell in Mpc (not Mpc/h).
    shell_redshifts : list of float
        Validated redshifts for each shell.

    Examples
    --------
    Basic usage with particle counts:

    >>> import numpy as np
    >>> from dorian.lensing import prepare_density_shells
    >>> # Create mock density maps (4 shells, nside=64)
    >>> nside = 64
    >>> npix = 12 * nside**2
    >>> density_maps = [np.random.poisson(100, npix) for _ in range(4)]
    >>> redshifts = [0.1, 0.2, 0.3, 0.4]
    >>> shells, distances, z_out = prepare_density_shells(
    ...     density_maps=density_maps,
    ...     redshifts=redshifts,
    ...     box_size=1000.0,        # Mpc/h
    ...     n_particles=256**3,
    ...     omega_m=0.3,
    ... )
    >>> len(shells)
    4

    With overdensity input:

    >>> delta_maps = [np.random.randn(npix) * 0.1 for _ in range(4)]
    >>> shells, distances, z_out = prepare_density_shells(
    ...     density_maps=delta_maps,
    ...     redshifts=redshifts,
    ...     box_size=1000.0,
    ...     n_particles=256**3,
    ...     omega_m=0.3,
    ...     unit='overdensity',
    ... )

    See Also
    --------
    raytrace_from_density : Combines preparation and ray-tracing in one call.
    dorian.raytracing.raytrace : Low-level ray-tracing function.
    """
    if omega_l is None:
        omega_l = 1.0 - omega_m

    if isinstance(box_size, (int, float)):
        box_size = (box_size, box_size, box_size)

    if nside is None:
        nside = hp.npix2nside(len(density_maps[0]))

    npix = hp.nside2npix(nside)

    rho_crit_h2 = 2.775e11
    rho_crit = rho_crit_h2 * h**2
    rho_matter = omega_m * rho_crit
    volume_box_mpc = (box_size[0] / h)**3
    particle_mass_msun = (rho_matter * volume_box_mpc) / n_particles
    particle_mass_dorian = (particle_mass_msun * h) / 1e10

    shells = []
    shell_distances = []
    shell_redshifts = []

    for density_map, z in zip(density_maps, redshifts):
        if hp.npix2nside(len(density_map)) != nside:
            density_map = hp.ud_grade(density_map, nside, power=-2)

        d_k = d_c(z=z, Omega_M=omega_m, Omega_L=omega_l)

        if unit == 'number_density':
            mass_per_pixel = density_map * particle_mass_dorian
        elif unit == 'overdensity':
            mean_density = n_particles / (box_size[0] * box_size[1] * box_size[2])
            number_density = (1 + density_map) * mean_density
            mass_per_pixel = number_density * particle_mass_dorian
        elif unit == 'mass':
            mass_per_pixel = density_map
        else:
            raise ValueError(f"Unknown unit type: {unit}. Must be 'number_density', 'overdensity', or 'mass'.")

        shells.append(np.asarray(mass_per_pixel, dtype=np.float64))
        shell_distances.append(float(d_k))
        shell_redshifts.append(float(z))

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
    unit='number_density',
    parallel_transport=True,
    lmax=0,
    nthreads=1,
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
    z_source : float
        Source redshift. Rays are traced from the observer to this redshift.
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

    unit : {'number_density', 'overdensity', 'mass'}, optional
        Input unit type for density maps:

        - ``'number_density'``: Particle count per pixel (default)
        - ``'overdensity'``: delta = rho/rho_mean - 1
        - ``'mass'``: Already in 10^10 M_sun/h per pixel

    parallel_transport : bool, optional
        Whether to apply parallel transport of the distortion matrix along
        geodesics. Recommended to keep True for accurate results. Default: True.
    lmax : int, optional
        Maximum multipole ell for spherical harmonic transforms.
        Default: ``3 * nside`` (sufficient for most applications).
    nthreads : int, optional
        Number of OpenMP threads for ``'nufft'`` interpolation. Default: 1.

    Returns
    -------
    results : dict
        Dictionary containing all ray-tracing outputs:

        - ``'convergence_born'`` : np.ndarray
            Born approximation convergence map (kappa). Shape: ``(npix,)``.
        - ``'convergence_raytraced'`` : np.ndarray
            Full ray-traced convergence from distortion matrix. Shape: ``(npix,)``.
        - ``'distortion_matrix'`` : np.ndarray
            Final Jacobian distortion matrix A. Shape: ``(2, 2, npix)``.
        - ``'ray_positions'`` : np.ndarray
            Final angular positions (theta, phi) of rays. Shape: ``(2, npix)``.
        - ``'initial_positions'`` : np.ndarray
            Initial ray positions (pixel centers). Shape: ``(2, npix)``.
        - ``'shell_info'`` : dict
            Metadata with keys: ``'redshifts'``, ``'distances'``,
            ``'n_shells_total'``, ``'n_shells_used'``.

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
    ...     nthreads=8,
    ... )

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

    shells, distances, redshifts_clean = prepare_density_shells(
        density_maps=density_maps,
        redshifts=redshifts,
        box_size=box_size,
        n_particles=n_particles,
        omega_m=omega_m,
        h=h,
        omega_l=omega_l,
        nside=nside,
        unit=unit,
    )

    if nside is None:
        nside = hp.npix2nside(len(shells[0]))

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
        nthreads=nthreads,
    )

    kappa_raytraced = 1.0 - 0.5 * (A_final[0, 0] + A_final[1, 1])

    n_used = sum(1 for z in redshifts_clean if z < z_source)

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
        }
    }
