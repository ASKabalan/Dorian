from .constants import M_sun_cgs, Mpc2cm, c_cgs, G_cgs
from .cosmology import d_c
from .gadget import load_all_headers, load_massmap, load_params
from .parallel_transport import get_rotation_angle_array, rotate_tensor_array
from .misc import print_logo
import numpy as np
import healpy as hp
from ducc0.sht import synthesis_general
from pathlib import Path
import time, h5py


def raytrace(
    simDir: str,
    z_s: float,
    interp: str = "ngp",
    outDir: str = "./",
    restart_file: str = "",
    lmax: int = 0,
    max_time_in_sec: int = 0,
    parallel_transport: bool = True,
    save_ray_positions: bool = False,
    nthreads: int = 1,
):
    """Routine for performing a ray tracing simulation.

    Parameters
    ----------
    simDir : str
        Path to the simulation folder.
    z_s : float
        Source redshift.
    interp : str
        Interpolation scheme to use: "ngp", "bilinear" and "nufft".
    outDir : str
        Path of the folder in which to save the maps.
    restart_file : str
        Path of the folder where the restart file is located. If provided, the
        simulation will restart from there. By default is a empty string,
        which means that the simulation starts from the innermost shell.
    lmax : float
        Maximum angular number for SHT computations (default 3 * Nside).
    max_time_in_sec : float
        Maximum time in seconds available for the simulation. If provided, the
        routine will take cake of writing restart files in the case the time is not
        sufficient for finishing the simulation.
    parallel_transport : bool
        Wheter to apply the parallel transport of the distorion matrix to the
        updated angular position of the ray at each plane. Setting this parameter
        true is recomended.
    save_ray_positions : bool
        Wheter to also save the angular position of the rays at each lens plane (the
        output becomes memory expensive).
    nthreads : int
        Number of OMP threads to use. At the moment this is only needed in the case
        of nufft interpolation.
    """
    t_begin = time.time()

    print_logo()
    print_initial_info(simDir, z_s, interp, outDir)
    print("Initializing data", flush=True)

    # Create output directory if it does not exist
    Path(f"{outDir}").mkdir(exist_ok=True)

    # Define factor to give physical units to the convergence
    kappa_fac = (1e10 * M_sun_cgs) * (1 / Mpc2cm) * 4 * np.pi * G_cgs / (c_cgs**2)

    # Load info about the relevant shells
    headers = load_all_headers(simDir)[::-1]
    sh_info = list(filter(lambda a: a["Redshift"] < z_s, headers))
    sh_ids = [a["Nmap"] for a in sh_info]  # shell ids
    print(f"The following shells will be used:\n{sh_ids}")
    npix = sh_info[0]["NpixTotal"]
    nside = hp.npix2nside(npix)
    # Compute the comoving distance of the k-th shell
    for k in range(len(sh_info)):
        sh_info[k]["ComDistMid"] = np.mean([sh_info[k]["ComDistStart"], sh_info[k]["ComDistEnd"]])

    # Compute comoving distance of the source
    params = load_params(simDir)
    Omega_M, Omega_L = params["Omega0"], params["OmegaLambda"]
    d_s = d_c(z=z_s, Omega_M=Omega_M, Omega_L=Omega_L)

    # Angular position of the rays when they reach the observer
    # We shoot one ray for every pixel center
    theta = np.array(hp.pix2ang(nside, np.arange(npix)))
    nrays = theta.shape[1]  # total number of rays
    # Angular position of the rays, dimensions are:
    # [k-th plane (previous, current), rows of beta (theta, phi), ray index]
    beta = np.zeros([2, 2, nrays])
    # Distorsion matrix, dimensions are:
    # [k-th plane (previous, current), rows of A, columns of A, ray index]
    A = np.zeros([2, 2, 2, nrays])
    # Convergence field in the Born approximation
    kappa_born = np.zeros([nrays])

    if not (restart_file):
        # Initialize quantities for the first lens plane
        beta[0] = theta
        beta[1] = theta
        for i in range(2):
            for j in range(2):
                A[0][i][j] = 1 if i == j else 0
                A[1][i][j] = 1 if i == j else 0
        sh_start = 0
    else:
        # Initialize quantities from the restart file
        with h5py.File(restart_file, "r") as f_restart:
            # TO DO check header
            sh_start = f_restart["Header"].attrs["sh_start"]
            beta[0] = np.array(f_restart["Ray_position"]["beta_0"])
            beta[1] = np.array(f_restart["Ray_position"]["beta_1"])
            A[0] = np.array(f_restart["Distortion_matrix"]["A_0"])
            A[1] = np.array(f_restart["Distortion_matrix"]["A_1"])
            kappa_born = np.array(f_restart["Distortion_matrix"]["kappa_born"])
        print(f"Loaded restart file from: {restart_file}")
        print(f"Iteration will restart from shell n. {sh_start}")

    # Define some constants to be used later in the SHT
    if lmax==0: 
        lmax = 3 * nside
    ell = np.arange(0, lmax + 1)

    # Iterate over the lens planes
    for k in range(sh_start, len(sh_ids)):
        t0 = time.time()
        print(f"*"*73, flush=True)
        print(f"Working on lens plane {k+1} of {len(sh_ids)}, MassMap n. {sh_ids[k]}")
        print("Computing convergence...", flush=True)
        # Define redshift and comoving distances of the k-th plane
        z_k = sh_info[k]["Redshift"]
        d_k = sh_info[k]["ComDistMid"]

        # Load the mass and compute Sigma ((1e10 M_sun/h)/sr)
        Sigma = load_massmap(simDir, sh_ids[k]) / (4 * np.pi / npix)
        # physical smoothing of 100 kpc
        # Sigma = hp.smoothing(Sigma, sigma=np.radians(1 / 60))
        Sigma_mean = np.mean(Sigma)

        # Compute convergence at the single lens plane
        kappa = kappa_fac * (1 + z_k) * (1 / d_k) * (Sigma - Sigma_mean)
        print(f"took {round(time.time()-t0,1)} s")

        # Compute quantities in spherical harmonics domain
        t0 = time.time()
        print("Computing quantities in spherical harmonics domain...", flush=True)
        kappa_lm = hp.map2alm(kappa, pol=False, lmax=lmax)
        alpha_lm = hp.almxfl(kappa_lm, -2 / (np.sqrt((ell * (ell + 1)))))
        f_l = -np.sqrt((ell + 2.0) * (ell - 1.0) / (ell * (ell + 1.0)))
        g_lm_E = hp.almxfl(kappa_lm, f_l)
        print(f"took {round(time.time()-t0,1)} s")

        # Evaluate alpha and U at desired angular positions: alpha(beta_k)
        t0 = time.time()
        print("Evaluating alpha and U at ray positions...", flush=True)

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

        print(f"took {round(time.time()-t0,1)} s")

        # Propagate every ray
        t0 = time.time()
        print("Propagating ray angular positions...", flush=True)
        
        # Compute distance of previous and next shell
        d_km1 = 0 if k==0 else sh_info[k-1]["ComDistMid"]
        d_kp1 = d_s if k == len(sh_ids) - 1 else sh_info[k+1]["ComDistMid"]
        # Compute distance-weighing pre-factors
        fac1 = d_k/d_kp1 * (d_kp1-d_km1)/(d_k-d_km1)
        fac2 = (d_kp1-d_k)/d_kp1

        for i in range(2):
            beta[0][i] = (1 - fac1) * beta[0][i] + fac1 * beta[1][i] - fac2 * alpha[i]

        # Update angular positions
        beta[[0, 1], ...] = beta[[1, 0], ...]

        # Make sure that all theta of beta[1] are in range [0, pi]
        # (only the poles need to be checked)
        check_theta_poles(beta[1])
        # Make sure that all phi of beta[1] are in range [0, 2*pi]
        beta[1][1] %= 2 * np.pi

        print(f"took {round(time.time()-t0,1)} s")

        # Propagate Distortion matrix for exery ray
        t0 = time.time()
        print("Propagating distortion matrix...", flush=True)

        for i in range(2):
            for j in range(2):
                A[0][i][j] = (
                    (1 - fac1) * A[0][i][j]
                    + fac1 * A[1][i][j]
                    - fac2 * (U[i][0] * A[1][0][j] + U[i][1] * A[1][1][j])
                )

        # Update distortion matrix
        A[[0, 1], ...] = A[[1, 0], ...]

        print(f"took {round(time.time()-t0,1)} s")

        # Parallel transport distortion matrix
        if parallel_transport:
            t0 = time.time()
            print("Parallel transporting distortion matrix...", flush=True)

            cospsi, sinpsi = get_rotation_angle_array(
                beta[0][0][:], beta[0][1][:], beta[1][0][:], beta[1][1][:]
            )
            A[0, :, :, :] = rotate_tensor_array(A[0, :, :, :], cospsi, sinpsi)
            A[1, :, :, :] = rotate_tensor_array(A[1, :, :, :], cospsi, sinpsi)

            print(f"took {round(time.time()-t0,1)} s")

        # Compute Born approximation convergence
        kappa_born += ((d_s - d_k) / d_s) * kappa

        # Eventually save ray positions
        if save_ray_positions:
            t0 = time.time()
            # Save ray posistions
            save_ray_positions_aux(
                simDir, outDir, k, params, beta[1], nside, interp, z_s
            )
            print(f"took {round(time.time()-t0,1)} s")

        # If there is not enough time for other 1.5 iterations, write restart file
        elapsed_sec = time.time() - t_begin
        estim_sec_per_iter = elapsed_sec / (k - sh_start + 1)
        print(f"Estimated seconds per iteration: {estim_sec_per_iter}")
        if max_time_in_sec and elapsed_sec + 1.5 * estim_sec_per_iter > max_time_in_sec:
            # Write restart file
            print("You are running out of time, wrtitng the restart file")
            fname_out = f"{outDir}restart_raytracing_shell{k}_z{z_s:.1f}_{interp}.hdf5"
            with h5py.File(fname_out, "a") as fout:
                write_header(fout, simDir, params["BoxSize"], z_s, nside)
                fout["Header"].attrs.create("sh_start", k + 1)
                grp_A = fout.create_group("Distortion_matrix")
                grp_A.create_dataset("A_0", data=A[0])
                grp_A.create_dataset("A_1", data=A[1])
                grp_A.create_dataset("kappa_born", data=kappa_born)
                grp_beta = fout.create_group("Ray_position")
                grp_beta.create_dataset("beta_0", data=beta[0])
                grp_beta.create_dataset("beta_1", data=beta[1])
            print(f"Wrote restart file to: {fname_out}")
            return

    # Save data
    print(f"*"*73, flush=True)
    fname_out = f"{outDir}raytracing_z{z_s:.1f}_{interp}.hdf5"
    with h5py.File(fname_out, "w") as fout:
        write_header(fout, simDir, params["BoxSize"], z_s, nside)
        grp_A = fout.create_group("Distortion_matrix")
        grp_A.create_dataset("Raytraced", data=A[1])
        grp_A.create_dataset("Kappa_born", data=kappa_born)
        grp_beta = fout.create_group("Ray_position")
        grp_beta.create_dataset("Beta", data=beta[1])
        grp_beta.create_dataset("Theta", data=theta)
    print(f"Saved data to: {fname_out}")

    print(f"*"*73, flush=True)
    print(f"Total time: {round(time.time()-t_begin)} s")
    print("Ray tracing finished, bye.")
    print(f"*"*73, flush=True)
    return


############################# AUXYLIARY FUNCTIONS #############################


def print_initial_info(simDir, z_s, interp, outDir):
    """Auxiliary function to write initial information about the run."""
    print(f"*"*73, flush=True)
    print(f"Input Directory:      {simDir}", flush=True)
    print(f"Source redshift:      {z_s}", flush=True)
    print(f"Interpolation method: {interp}", flush=True)
    print(f"Output directory:     {outDir}", flush=True)
    print(f"*"*73, flush=True)


def write_header(fout, simDir, boxSize, z_s, nside):
    """Auxiliary function to write the header."""
    grp_header = fout.create_group("Header")
    grp_header.attrs.create("SimDir", simDir)
    grp_header.attrs.create("BoxSize", boxSize)
    grp_header.attrs.create("Source_redshift", z_s)
    grp_header.attrs.create("Nside", nside)
    return


def get_val(m_list, theta, phi, interp):
    """Auxiliary function for interpolation.
    m_list : list of 1-D arrays
        list of healpix maps to interpolate over.
    theta : 1-D array
        co latitudes of points where to interpolate.
    phi : 1-D array
        longitudes of points where to interpolate.
    interp : string
        interpolation method.
    """
    nside = hp.npix2nside(len(m_list[0]))
    if interp == "ngp":
        idx = hp.ang2pix(nside, theta, phi)
        return [m[idx] for m in m_list]
    if interp == "bilinear":
        p, w = hp.get_interp_weights(nside, theta, phi)
        return [np.sum(m[p] * w, 0) for m in m_list]


def get_val_nufft(alm, theta, phi, spin, lmax, nthreads):
    """Auxiliary function for interpolation.
    alm : np array
        alm coefficients.
    theta : 1-D array
        co latitudes of points where to interpolate.
    phi : 1-D array
        longitudes of points where to interpolate.
    spin : int
        spin of the field.
    lmax : int
        maximum multipole number
    nthreads : int
        number of OpenMP threads
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
    """Routine for checking that the theta coordinate is in the range [0, pi].
    We look for such extreme values only for the first and last rays, assuming
    that these are HEALPix ring ordered. I.e. we check the 1% of the rays with 
    most extreme latitudes.

    Parameters
    ----------
    coords : 2D numpy array (shape: [2, nrays])
        Spherical coordinates: theta and phi respectively in the first and second
        column. One row for every ray
    """
    n_check = int(0.005 * len(coords[:]))
    # select both poles
    coords_pole_north = coords[:, :n_check]
    coords_pole_south = coords[:, -n_check:]
    for coords_pole in [coords_pole_north, coords_pole_south]:
        # take care of negative theta
        idx = np.where(coords_pole[0] < 0)
        coords_pole[0, idx] = -coords_pole[0, idx]
        # take care of theta that are greater than pi
        idx = np.where(coords_pole[0] > np.pi)
        coords_pole[0, idx] = 2 * np.pi - coords_pole[0, idx]
        # make sure phi is not greater than 2*pi
        coords_pole[1, idx] += np.pi
        coords_pole[1, idx] %= 2 * np.pi
    # reassign beta
    coords[:, :n_check] = coords_pole_north
    coords[:, -n_check:] = coords_pole_south


def save_ray_positions_aux(
    simDir, outDir, k_shell, params, beta_1, nside, interp, z_s
):
    """Auxiliary function to save ray positions"""
    print("Saving ray positions...")
    fname_out = f"{outDir}ray_position_shell{k_shell}_z{z_s:.1f}_{interp}.hdf5"
    with h5py.File(fname_out, "a") as fout:
        write_header(fout, simDir, params["BoxSize"], z_s, nside)
        fout["Header"].attrs.create("shell_number", k_shell)
        grp_beta = fout.create_group("Ray_position")
        grp_beta.create_dataset("beta", data=beta_1)
    print(f"Wrote ray positions to: {fname_out}")
