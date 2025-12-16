from .constants import M_sun_cgs, Mpc2cm, c_cgs, G_cgs
from .cosmology import d_c
from .parallel_transport import get_rotation_angle_array, rotate_tensor_array
from .misc import print_logo
import numpy as np
import healpy as hp
from ducc0.sht import synthesis_general
import time


def raytrace(
    shells,
    z_s,
    omega_m,
    omega_l,
    nside,
    shell_redshifts,
    shell_distances,
    interp="bilinear",
    lmax=0,
    parallel_transport=True,
    nthreads=1,
):
    """
    Perform full-sky gravitational lensing ray-tracing through multiple lens planes.

    This is the core ray-tracing algorithm in Dorian. It propagates light rays
    backwards from the observer through a series of mass shells, computing
    deflections and the Jacobian distortion matrix at each step.

    Parameters
    ----------
    shells : list of np.ndarray
        List of HEALPix mass maps for each shell. Each map must contain mass
        per pixel in units of 10^10 M_sun/h. Use ``prepare_density_shells()``
        to convert from other formats. Maps should be in RING ordering.
    z_s : float
        Source redshift. Only shells with ``z < z_s`` will contribute.
    omega_m : float
        Matter density parameter (Omega_m).
    omega_l : float
        Dark energy density parameter (Omega_Lambda).
    nside : int
        HEALPix NSIDE parameter defining the angular resolution.
        Number of pixels = ``12 * nside**2``.
    shell_redshifts : list of float
        Redshift of each shell, same length as ``shells``.
    shell_distances : list of float
        Comoving distance to each shell in Mpc (not Mpc/h).
    interp : {'bilinear', 'ngp', 'nufft'}, optional
        Interpolation method for sampling fields at ray positions:

        - ``'bilinear'``: Bilinear interpolation (default, recommended)
        - ``'ngp'``: Nearest grid point (fastest, lowest accuracy)
        - ``'nufft'``: Non-uniform FFT via ducc0 (highest accuracy)

    lmax : int, optional
        Maximum multipole ell for spherical harmonic transforms.
        Default: ``3 * nside`` when set to 0.
    parallel_transport : bool, optional
        Whether to parallel transport the distortion matrix along geodesics
        when rays are deflected. Recommended True for accurate results.
        Default: True.
    nthreads : int, optional
        Number of OpenMP threads for ``'nufft'`` interpolation. Default: 1.

    Returns
    -------
    kappa_born : np.ndarray
        Born approximation convergence map. Shape: ``(npix,)``.
    A_final : np.ndarray
        Final Jacobian distortion matrix. Shape: ``(2, 2, npix)``.
        Components: A[0,0]=A_11, A[0,1]=A_12, A[1,0]=A_21, A[1,1]=A_22.
    beta_final : np.ndarray
        Final angular positions (theta, phi) of rays. Shape: ``(2, npix)``.
        theta in [0, pi], phi in [0, 2*pi].
    theta : np.ndarray
        Initial ray positions (HEALPix pixel centers). Shape: ``(2, npix)``.

    Notes
    -----
    The algorithm iterates through shells from observer outward:

    1. Compute surface density Sigma and convergence kappa for each shell
    2. Transform kappa to spherical harmonics to get deflection angles alpha
    3. Interpolate alpha at current ray positions (post-Born correction)
    4. Update ray positions using the multi-plane lens equation
    5. Update distortion matrix A with the tidal tensor U
    6. Optionally parallel transport A to the new ray direction
    7. Accumulate Born approximation convergence

    The distortion matrix A relates source to image coordinates:
    ``d(beta)/d(theta) = A``, initialized to identity.

    Lensing observables from A:
        - Convergence: ``kappa = 1 - 0.5 * (A[0,0] + A[1,1])``
        - Shear: ``gamma1 = 0.5 * (A[0,0] - A[1,1])``, ``gamma2 = A[0,1]``

    Examples
    --------
    Low-level usage (prefer ``raytrace_from_density`` for convenience):

    >>> from dorian.raytracing import raytrace
    >>> from dorian.lensing import prepare_density_shells
    >>> # Prepare shells first
    >>> shells, distances, redshifts = prepare_density_shells(...)
    >>> # Run ray-tracing
    >>> kappa_born, A, beta, theta = raytrace(
    ...     shells=shells,
    ...     z_s=1.0,
    ...     omega_m=0.3,
    ...     omega_l=0.7,
    ...     nside=512,
    ...     shell_redshifts=redshifts,
    ...     shell_distances=distances,
    ... )
    >>> # Compute ray-traced convergence
    >>> kappa_rt = 1.0 - 0.5 * (A[0, 0] + A[1, 1])

    See Also
    --------
    dorian.lensing.raytrace_from_density : High-level interface (recommended).
    dorian.lensing.prepare_density_shells : Convert density maps to mass format.
    """
    t_begin = time.time()

    print_logo()

    kappa_fac = (1e10 * M_sun_cgs) * (1 / Mpc2cm) * 4 * np.pi * G_cgs / (c_cgs**2)

    contributing_shells = []
    for i, (shell, z_k, d_k) in enumerate(zip(shells, shell_redshifts, shell_distances)):
        if z_k < z_s:
            contributing_shells.append({
                'shell_data': shell,
                'redshift': z_k,
                'distance': d_k,
                'index': i
            })

    print(f"Using {len(contributing_shells)} shells out of {len(shells)} total")

    if len(contributing_shells) == 0:
        raise ValueError(f"No shells found with z < z_s ({z_s}). Check your shell redshifts.")

    print(f"Shell redshift range: {contributing_shells[0]['redshift']:.3f} to {contributing_shells[-1]['redshift']:.3f}")

    npix = hp.nside2npix(nside)

    d_s = d_c(z=z_s, Omega_M=omega_m, Omega_L=omega_l)

    theta = np.array(hp.pix2ang(nside, np.arange(npix)))
    nrays = theta.shape[1]
    beta = np.zeros([2, 2, nrays])
    A = np.zeros([2, 2, 2, nrays])
    kappa_born = np.zeros([nrays])

    beta[0] = theta
    beta[1] = theta
    for i in range(2):
        for j in range(2):
            A[0][i][j] = 1 if i == j else 0
            A[1][i][j] = 1 if i == j else 0
    sh_start = 0

    if lmax==0:
        lmax = 3 * nside
    ell = np.arange(0, lmax + 1)

    for k in range(sh_start, len(contributing_shells)):
        t0 = time.time()
        shell_info = contributing_shells[k]
        z_k = shell_info['redshift']
        d_k = shell_info['distance']
        shell_data = shell_info['shell_data']

        Sigma = shell_data / (4 * np.pi / npix)
        Sigma_mean = np.mean(Sigma)

        kappa = kappa_fac * (1 + z_k) * (1 / d_k) * (Sigma - Sigma_mean)

        kappa_lm = hp.map2alm(kappa, pol=False, lmax=lmax)
        alpha_lm = hp.almxfl(kappa_lm, -2 / (np.sqrt((ell * (ell + 1)))))
        f_l = -np.sqrt((ell + 2.0) * (ell - 1.0) / (ell * (ell + 1.0)))
        g_lm_E = hp.almxfl(kappa_lm, f_l)

        if interp in ["ngp", "bilinear"]:
            alpha = hp.alm2map_spin(
                [alpha_lm, np.zeros_like(alpha_lm)], nside=nside, spin=1, lmax=lmax
            )
            alpha = get_val(alpha, beta[1][0], beta[1][1], interp=interp)

            g1, g2 = hp.alm2map_spin(
                [g_lm_E, np.zeros_like(g_lm_E)], nside=nside, spin=2, lmax=lmax
            )
            U = np.zeros([2, 2, nrays])
            U[0][0] = kappa + g1
            U[1][0] = g2
            U[0][1] = U[1][0]
            U[1][1] = kappa - g1
            U[0, 0], U[0, 1], U[1, 1] = get_val(
                [U[0, 0], U[0, 1], U[1, 1]], beta[1][0], beta[1][1], interp=interp
            )
            U[1, 0] = U[0, 1]

        elif interp == "nufft":
            alpha = get_val_nufft(
                alpha_lm, beta[1][0], beta[1][1], spin=1, lmax=lmax, nthreads=nthreads
            )
            g1, g2 = get_val_nufft(
                g_lm_E, beta[1][0], beta[1][1], spin=2, lmax=lmax, nthreads=nthreads
            )
            kappa_nufft = get_val_nufft(
                kappa_lm, beta[1][0], beta[1][1], spin=0, lmax=lmax, nthreads=nthreads
            )[0]

            U = np.zeros([2, 2, nrays])
            U[0][0] = kappa_nufft + g1
            U[1][0] = g2
            U[0][1] = U[1][0]
            U[1][1] = kappa_nufft - g1

        d_km1 = 0 if k==0 else contributing_shells[k-1]['distance']
        d_kp1 = d_s if k == len(contributing_shells) - 1 else contributing_shells[k+1]['distance']
        fac1 = d_k/d_kp1 * (d_kp1-d_km1)/(d_k-d_km1)
        fac2 = (d_kp1-d_k)/d_kp1

        for i in range(2):
            beta[0][i] = (1 - fac1) * beta[0][i] + fac1 * beta[1][i] - fac2 * alpha[i]

        beta[[0, 1], ...] = beta[[1, 0], ...]

        check_theta_poles(beta[1])
        beta[1][1] %= 2 * np.pi

        for i in range(2):
            for j in range(2):
                A[0][i][j] = (
                    (1 - fac1) * A[0][i][j]
                    + fac1 * A[1][i][j]
                    - fac2 * (U[i][0] * A[1][0][j] + U[i][1] * A[1][1][j])
                )

        A[[0, 1], ...] = A[[1, 0], ...]

        if parallel_transport:

            cospsi, sinpsi = get_rotation_angle_array(
                beta[0][0][:], beta[0][1][:], beta[1][0][:], beta[1][1][:]
            )
            A[0, :, :, :] = rotate_tensor_array(A[0, :, :, :], cospsi, sinpsi)
            A[1, :, :, :] = rotate_tensor_array(A[1, :, :, :], cospsi, sinpsi)

        kappa_born += ((d_s - d_k) / d_s) * kappa

    print(f"*"*73, flush=True)
    print(f"Total time: {round(time.time()-t_begin)} s")
    print("Ray tracing finished, bye.")
    print(f"*"*73, flush=True)
    return kappa_born, A[1], beta[1], theta


def get_val(m_list, theta, phi, interp):
    """
    Interpolate HEALPix maps at arbitrary angular positions.

    Parameters
    ----------
    m_list : list of np.ndarray
        List of HEALPix maps to interpolate. All maps must have the same NSIDE.
    theta : np.ndarray
        Co-latitude coordinates in radians, range [0, pi]. Shape: ``(npoints,)``.
    phi : np.ndarray
        Longitude coordinates in radians, range [0, 2*pi]. Shape: ``(npoints,)``.
    interp : {'ngp', 'bilinear'}
        Interpolation method:

        - ``'ngp'``: Nearest grid point (fastest)
        - ``'bilinear'``: Bilinear interpolation using 4 nearest pixels

    Returns
    -------
    list of np.ndarray
        Interpolated values for each input map. Each array has shape ``(npoints,)``.
    """
    nside = hp.npix2nside(len(m_list[0]))
    if interp == "ngp":
        idx = hp.ang2pix(nside, theta, phi)
        return [m[idx] for m in m_list]
    if interp == "bilinear":
        p, w = hp.get_interp_weights(nside, theta, phi)
        return [np.sum(m[p] * w, 0) for m in m_list]


def get_val_nufft(alm, theta, phi, spin, lmax, nthreads):
    """
    Evaluate spherical harmonic expansion at arbitrary positions using NUFFT.

    Uses ducc0's ``synthesis_general`` for high-accuracy non-uniform evaluation
    of spin-weighted spherical harmonic transforms.

    Parameters
    ----------
    alm : np.ndarray
        Spherical harmonic coefficients in healpy's indexing convention.
    theta : np.ndarray
        Co-latitude coordinates in radians, range [0, pi]. Shape: ``(npoints,)``.
    phi : np.ndarray
        Longitude coordinates in radians, range [0, 2*pi]. Shape: ``(npoints,)``.
    spin : int
        Spin weight of the field (0 for scalars, 1 for vectors, 2 for tensors).
    lmax : int
        Maximum multipole ell in the expansion.
    nthreads : int
        Number of OpenMP threads for parallel computation.

    Returns
    -------
    np.ndarray
        Evaluated field values at the specified positions.
        For spin > 0, returns array with shape ``(2, npoints)`` for the two
        spin components.
    """
    if spin == 0:
        alm2 = alm.reshape((1, -1))
    elif spin > 0:
        alm2 = np.zeros((2, alm.shape[0]), dtype=alm.dtype)
        alm2[0] = alm
    return synthesis_general(
        alm=alm2, spin=spin, lmax=lmax, loc=np.vstack([theta, phi]).T, nthreads=nthreads
    )


def check_theta_poles(coords):
    """
    Ensure theta coordinates remain within valid range [0, pi] near poles.

    After ray deflection, some rays near the poles may have theta values
    outside the valid range. This function corrects them by reflecting
    across the poles and adjusting phi accordingly.

    Only checks rays near the poles (0.5% at each end) assuming HEALPix
    RING ordering where polar rays are at the start and end of the array.

    Parameters
    ----------
    coords : np.ndarray
        Ray coordinates with shape ``(2, nrays)``. First row is theta
        (co-latitude), second row is phi (longitude). Modified in-place.

    Notes
    -----
    Correction rules:

    - If theta < 0: reflect to -theta, shift phi by pi
    - If theta > pi: reflect to 2*pi - theta, shift phi by pi
    """
    n_check = int(0.005 * len(coords[:]))
    coords_pole_north = coords[:, :n_check]
    coords_pole_south = coords[:, -n_check:]
    for coords_pole in [coords_pole_north, coords_pole_south]:
        idx = np.where(coords_pole[0] < 0)
        coords_pole[0, idx] = -coords_pole[0, idx]
        idx = np.where(coords_pole[0] > np.pi)
        coords_pole[0, idx] = 2 * np.pi - coords_pole[0, idx]
        coords_pole[1, idx] += np.pi
        coords_pole[1, idx] %= 2 * np.pi
    coords[:, :n_check] = coords_pole_north
    coords[:, -n_check:] = coords_pole_south
