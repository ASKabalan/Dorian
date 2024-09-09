import numpy as np
import healpy as hp
import os
from scipy import stats, interpolate
from .cosmology import d_c, z_from_d_c
from .constants import c_cgs, Mpc2cm, Mpc2km


def load_Pk_gadget(simDir, nsnap, nbins=None, logbins=True, getz=False, fout=None):
    """Loads the 3D matter power spectrum from a Gadget-4 simulation.

    Parameters
    ----------
    simDir : str
        Path to the simulation output folder.
    nsnap : int
        Number of the shapshot.
    nbins : int
        Number of k bins.
    logbins : bool
        Whether the k bins are log spaced or not.
    getz : bool
        Returns in addition the snapshot redshift.
    fout : str
        Path to the folder in which to save the data.

    Returns
    -------
    kvals : array
        Array containing the values of the wavenumber k on which the
        bins are centered.
    Pk : array
        Array containing P(k).
    z: float
        The z of the power spectrum (returned only if getz is true).
    """
    # Look for the file
    found = False
    for file in os.listdir(simDir + "/powerspecs"):
        fname = os.fsdecode(file)
        if fname.endswith(f"_{nsnap:03d}.txt"):
            found = True
            break
    if not (found):
        print("File not found!")
        return
    # Read the file
    fpath = simDir + "/powerspecs/" + fname
    f = open(fpath)
    lines = f.readlines()
    nbins_file = int(lines[1])
    kvals, Pk = [], []
    z = 1 / float(lines[0]) - 1
    for i in range(5, nbins_file + 5):
        idx = [j for j, letter in enumerate(lines[i]) if letter == " "]
        k = float(lines[i][: idx[0]])
        kvals.append(k)
        delta = float(lines[i][idx[0] : idx[1]])
        Pk.append(2 * (np.pi) ** 2 * delta / k**3)
    # Create bins for k
    if nbins:
        if logbins:
            kbins = np.logspace(np.log10(kvals[0]), np.log10(kvals[-1]), nbins + 1)
        else:
            kbins = np.linspace(kvals[0], kvals[-1], nbins + 1)
        Pk, _, _ = stats.binned_statistic(kvals, Pk, statistic="mean", bins=kbins)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
    # Eventually save Pk in numpy format
    if fout:
        print(f"Saved Pk to {fout}_Pk")
        np.save(fout + "_Pk", np.stack((kvals, Pk), axis=-1))
    if getz:
        return kvals, Pk, z
    else:
        return kvals, Pk


def Pk_2D(m, lside, nbins=20, logbins=True, zero_padding=False, fout=None):
    """Computes the P(k) of a 2D density field. This field is given as input in the
    square matrix m, which must be a np array.

    Parameters
    ----------
    m : 2D array
        Square map on which to compute the power spectrum.
    lside : float
        Physical lenght of the side of the map.
    nbins : int
        Number of k bins.
    logbins : bool
        Whether the l bins are log spaced or not.
    zero_padding : bool:
        Whether to apply zero padding or not
    fout : str
        Path to the folder in which to save the data.

    Returns
    -------
    kvals : array
        Array containing the values of the wavenumber k on which the
        bins are centered.
    Abins : array
        Array containing P(k).
    """
    n = m.shape[0] if not (zero_padding) else 2 * m.shape[0]
    lside = lside if not (zero_padding) else 2 * lside
    # Compute fundamental frequency
    kF = 2.0 * np.pi / lside
    # Compute the Fourier transform of m
    if not (zero_padding):
        fourier_m = np.fft.fftn(m)
    else:
        m_pad = np.zeros([n, n], dtype=float)
        m_pad[: m.shape[0], : m.shape[0]] += m
        fourier_m = np.fft.fftn(m_pad)
    # Compute the amplitude of the Fourier components
    fourier_amp = np.abs(fourier_m) ** 2
    # Construct the wave vector array
    kfreq = np.fft.fftfreq(n) * n
    kfreq2D = np.meshgrid(kfreq, kfreq)
    # We are interested in the norm of the wave vectors
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    # We can now flatten the 2D arrays
    knrm, fourier_amp = knrm.flatten(), fourier_amp.flatten()
    # Create bins for k
    k_start = 0.5 if not (zero_padding) else 1.0
    if logbins:
        kbins = np.logspace(np.log10(k_start), np.log10(n // 2 + 1), nbins + 1)
    else:
        kbins = np.linspace(k_start, n // 2 + 1, nbins + 1)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(
        knrm, fourier_amp, statistic="mean", bins=kbins
    )
    if zero_padding:
        Abins *= 4
    # Give units
    for i in range(len(kvals)):
        kvals[i] = (kvals[i]) * kF
        Abins[i] = (Abins[i]) * (lside / n**2) ** 2
    # Eventually save Pk in numpy format
    if fout:
        print(f"Saved Pk to {fout}_Pk")
        np.save(fout + "_Pk", np.stack((kvals, Abins), axis=-1))
    return np.array([kvals, Abins])


def C_ell(m, lmax=None, nest=False, nbins=None, logbins=True, fout=None):
    """Computes the angular power spectrum (known as C(l)) of the map m through the
    healpy anafast routine.

    Parameters
    ----------
    m : array
        HEALPix map on which to compute the power spectrum.
    lmax : float
        Maximum l of the power spectrum (default: 3*nside-1).
    nest : bool
        Whether the map is nest ordered or not.
    nbins : int
        Number of l bins.
    logbins : bool
        Whether the l bins are log spaced or not.
    fout : str
        Path to the folder in which to save the data.

    Returns
    -------
    l : array
        Array containing the values of l.
    cl : array
        Array containing C(l).
    """
    rho = m
    # Map must be ring ordered
    if nest:
        rho = hp.reorder(rho, n2r=True)
    # Compute the power spectrum
    if lmax == None:
        Cl = hp.anafast(rho, use_weights=True)
    else:
        Cl = hp.anafast(rho, use_weights=True, lmax=lmax)
    ll = np.arange(len(Cl))
    # Eventually rebin
    if nbins:
        if logbins:
            lbins = np.logspace(np.log10(1), np.log10(ll[-1]), nbins + 1)
        else:
            lbins = np.linspace(ll[0], ll[-1], nbins + 1)
        Cl, _, _ = stats.binned_statistic(ll, Cl, statistic="mean", bins=lbins)
        ll = 0.5 * (lbins[1:] + lbins[:-1])
    # Eventually save C(l) in numpy format
    if fout:
        print(f"Saved Cl to {fout}_Cl")
        np.save(fout + "_Cl", np.stack((ll, Cl), axis=-1))
    return ll, Cl