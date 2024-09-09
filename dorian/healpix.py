import numpy as np
import healpy as hp
from functools import partial


def extract_region(m, lon, lat, phi=0, alpha=5, n=1024, fout=None, nest=False):
    """Extracts a n x n matrix containing the values of the map m inside a square patch.
    The square patch is centered at (lon, lat) and has a side of alpha degrees.

    Parameters
    ----------
    m : array-like
        HEALPix map from which extract the region.
    lon : array-like or int
        Longitude(s) of the maps center(s), in degrees.
    lat : array-like or int
        Latitude(s) of the maps center(s), in degrees.
    phi : array-like or int
        Rotation angle, in degrees.
    alpha : float
        Angular size of the maps, in degrees.
    n : int
        Number of pixels per side of the region.
    fout : str
        Path of the folder which to save the output.
    nest : bool
        Whether the angular maps are nest ordered or not.

    Returns
    -------
    2D array
        The desired region.
    """
    # Compute region
    nside = hp.npix2nside(len(m))
    proj = hp.projector.GnomonicProj(
        rot=(lon, lat, phi), xsize=n, reso=(alpha * 60) / n
    )
    region = proj.projmap(m, vec2pix_func=partial(hp.vec2pix, nside))
    # Eventually save region in numpy format
    if fout:
        print(f"Saved region to {fout}")
        np.save(fout, region)
    return region


def smooth_top_hat(m, radius, nest=False):
    """Smooth an HEALPix map with a top hat filter with circular shape with
    given radius.

    Parameters
    ----------
    m : array-like
        HEALPix map to smooth.
    radius : float
        Radius of the filter (in radians).
    nest : bool
        Whether the angular maps are nest ordered or not.

    Returns
    -------
    array-like
        The smoothed map.
    """
    npix = m.shape[0]
    nside = hp.npix2nside(npix)
    m_out = np.zeros(npix, dtype=m.dtype)
    for ipix in range(npix):
        theta, phi = hp.pix2ang(nside, ipix, nest=nest)
        vec = hp.ang2vec(theta, phi)
        disc = hp.query_disc(nside, vec, radius, nest=nest)
        m_out[ipix] = np.mean(m[disc])
    return m_out


def rotate_map(m, rot_theta, rot_phi, nest=False):
    """Take a healpix map m and returns another map healpix rotated
    by the (theta, phi) given.

    Parameters
    ----------
    m : array-like
        HEALPix map to rotate.
    rot_theta : float
        Rotation angle theta in radians.
    rot_phi : float
        Rotation angle phi in radians.
        nest : bool
        Whether the angular maps are nest ordered or not.

    Returns
    -------
    array-like
        Rotated map.
    """
    nside = hp.npix2nside(len(m))
    # Get theta, phi for non-rotated map
    t, p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))  # theta, phi
    # Define a rotator
    r = hp.Rotator(deg=False, rot=[rot_phi, rot_theta])
    # Get theta, phi under rotated co-ordinates
    trot, prot = r(t, p)
    # Interpolate map onto these co-ordinates
    rot_map = hp.get_interp_val(m, trot, prot, nest=nest)
    return rot_map


def ud_grade_interp(map_in, nside_out, nest=False, interpolation="bilinear"):
    """Up or downgrades an healpix map to a map nside = nside_out.
    The values in the output map are computed via either NGP or bilinear
    interpolation.

    Parameters
    ----------
    map_in : array-like
        HEALPix map to up/down-grade.
    nside_out : int
        NSIDE of the output map.
    nest : bool
        Whether the angular maps are nest ordered or not.
    interp : str
        Interpolation scheme to use: "ngp" or "bilinear".


    Returns
    -------
    array-like
        Up/downgraded map.
    """
    nside_in = hp.npix2nside(map_in.shape[0])
    npix_out, npix_in = hp.nside2npix(nside_out), hp.nside2npix(nside_in)
    map_out = np.zeros(npix_out, dtype=map_in.dtype)
    theta, phi = hp.pix2ang(nside_out, np.array(range(npix_out)), nest=nest)
    if interpolation == "bilinear":
        map_out = hp.get_interp_val(map_in, theta, phi, nest=nest)
    elif interpolation == "ngp":
        map_out = map_in[hp.ang2pix(nside_in, theta, phi, nest=nest)]
    else:
        print("interpolation method not valid")
        return
    map_out /= npix_out / npix_in
    return map_out


def get_rings_idx(nside, get_colatitude=False):
    """Computes the starting index of each HEALPix ring, given a certain Nside.

    Parameters
    ----------
    nside : int
        HEALPix Nside parameter.
    get_colatitude : bool
        Wheter to also get the co-latitude of the respective rings.

    Returns
    -------
    array-like
        Indices of the rings.
    """
    rings_idx = []
    npix = hp.nside2npix(nside)
    ncap = 2 * nside * (nside - 1)
    # north polar cap
    for i_z in range(1, nside):
        i_ring = i_z
        ipix = 2 * i_ring * (i_ring - 1)
        rings_idx.append(ipix)
    # equatorial region
    for i_z in range(nside, 3 * nside + 1):
        i_ring = i_z - nside + 1
        ipix = ncap + 4 * nside * (i_ring - 1)
        rings_idx.append(ipix)
    # south polar cap
    for i_z in range(3 * nside + 1, 4 * nside):
        i_ring = 4 * nside - i_z
        ipix = npix - 2 * i_ring * (i_ring + 1)
        rings_idx.append(ipix)
    rings_idx = np.array(sorted(rings_idx))
    if not (get_colatitude):
        return rings_idx
    else:
        colatitude = hp.pix2ang(nside, rings_idx)
        return rings_idx, colatitude[0]


def shot_noise(nshot, nside):
    """Fills the healpix map "m" (with the chosen nside) with a number of shots
    drawn from a uniform distribution, so to obtain Poisson noise.

    Parameters
    ----------
    nshot : int
        Number of shots to draw.
    nside : int
        HEALPix Nside parameter.

    Returns
    -------
    array-like
        HEALPix noise map'.
    """
    npix = 12 * nside**2
    m = np.zeros(npix)
    for i in range(nshot):
        m[np.random.randint(0, npix)] += 1
    return m


def IndexToDecRa(index, nside):
    """Converts HEALPix index to corresponding Re and Dec.

    Parameters
    ----------
    index : int
        HEALPix index.
    nside : int
        HEALPix Nside.

    Returns
    -------
    Float
        Declination.
    Float
        RIght ascension.
    """
    theta, phi = hp.pixelfunc.pix2ang(nside, index)
    return -np.degrees(theta - np.pi / 2.0), np.degrees(np.pi * 2.0 - phi)


def alm2map_der2(alm, nside, lmax=None, mmax=None):
    """Computes a Healpix map and its second derivatives given the alm.

    The alm are given as a complex array. You can specify lmax
    and mmax, or they will be computed from array size (assuming
    lmax==mmax).
    
    The notations of kappa and gamma are borrowed from the weak lensing
    formalism. See e.g. eq (7, 8, 9, 25, 26) of 
    (Castro et al. 2005, Phys. Rev. D, 72, 023516)

    Parameters
    ----------
    alm : array, complex
      A complex array of alm. Size must be of the form mmax(lmax-mmax+1)/2+lmax
    nside : int
      The nside of the output map.
    lmax : None or int, optional
      Explicitly define lmax (needed if mmax!=lmax)
    mmax : None or int, optional
      Explicitly define mmax (needed if mmax!=lmax)

    Returns
    -------
    m, d_theta^2, d_theta*d_phi , d_phi^2 : tuple of arrays
      The maps correponding to alm, and its second derivatives with respect 
      to theta and phi. d_theta*d_phi is already divided by sin(theta) and
      d_phi^2 is already divided by sin^2(theta)
    """
    if lmax is None: 
        lmax = 3 * nside
        mmax = lmax
    m = hp.alm2map(alm, nside=nside, lmax=lmax, mmax=mmax)
    ell = np.arange(0, lmax + 1)    
    f_l = np.sqrt((ell + 2.0) * (ell - 1.0) / (ell * (ell + 1.0)))
    kappa_lm = hp.almxfl(alm, -0.5*(ell * (ell + 1)))
    g_lm = hp.almxfl(kappa_lm, f_l)
    kappa = hp.alm2map(kappa_lm, nside=nside, lmax=lmax, mmax=mmax)
    g1, g2 = hp.alm2map_spin(
        [g_lm, np.zeros_like(g_lm)], nside=nside, spin=2, lmax=lmax, mmax=mmax
    )
    return m, kappa + g1, g2, kappa - g1