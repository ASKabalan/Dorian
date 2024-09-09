import numpy as np
from scipy import ndimage


def smooth_region(m, sigma, lside, fout=None):
    """Convolves a square map with a gaussian kernel. Note that sigma and lside
    must be given in the same units.

    Parameters
    ----------
    m : 2D array
        Image on which to smooth.
    sigma : float
        Standard deviation of the kernel.
    lside : float
        Physical lenght of the side of the image.
    fout : str
        Path of the folder which to save the output.

    Returns
    -------
    2D array
        The smoothed map.
    """
    sigma_0 = sigma / (lside / m.shape[0])
    m_smooth = ndimage.filters.gaussian_filter(m, sigma_0, mode="constant")
    # Eventually save smoothed region in numpy format
    if fout:
        print(f"Saved smoothed region to {fout}_smoothed")
        np.save(fout + "_smoothed", m_smooth)
    return m_smooth


def add_gaussian_noise(m, sigma_e, n_gal, A_pix, fout=None):
    """Adds gaussian shape noise to a square map, as modeled in eq. (7) of
    Lee+ 22 https://arxiv.org/abs/2201.08320.

    Parameters
    ----------
    m : 2D-array or HEALPix map
        Image on which to add the gaussian noise.
    sigma_e : float
        Intrinsic ellipticity of galaxies.
    n_gal : float
        Surface number density of lensed galaxies in arcmin^-2.
    sigma_e : float
        Intrinsical ellipticity of galaxies.
    A_pix : float
        Solid angle of a pixel in arcmin^2.
    fout : str
        Path of the folder which to save the output.

    Returns
    -------
    array-like
        Image with gaussian noise added.
    """
    sigma = sigma_e / np.sqrt(2 * n_gal * A_pix)
    noise = np.random.normal(0, sigma, m.shape)
    m_noise = m + noise
    # Eventually save region with noise in numpy format
    if fout:
        print(f"Saved region with noise to {fout}_noise")
        np.save(fout + "_noise", m_noise)
    return m_noise


def fibonacci_sphere(n=1000, lonlat=True):
    """Distribute n points on a sphere with the Fibonacci algorithm, returns
    the positions of such points.
    More on the algorithm at:
    https://extremelearning.com.au/evenly-distributing-points-on-a-sphere/

    Parameters
    ----------
    n : int
        Number of points to place on the sphere.
    lonlat : bool
        wheter the output has to be as longitude and latitude in degrees or
        colatitude and longitude in radians.

    Returns
    -------
    2D array
        Array with shape n x 2, each row representing a point.
    """
    goldenRatio = (1 + 5**0.5) / 2
    i = np.arange(0, n)
    phi = (2 * np.pi * i / goldenRatio) % (2 * np.pi)
    theta = np.arccos(1 - 2 * (i + 0.5) / n)
    if lonlat:
        coords = np.stack((np.degrees(phi), np.degrees(theta - np.pi / 2)), axis=-1)
    else:
        coords = np.stack((theta, phi), axis=-1)
    return coords


def print_logo():
    logo = """
    ▐▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▌
    ▐                                                               ▌
    ▐   ██████████                       ███                        ▌
    ▐  ░░███░░░░███                     ░░░                         ▌
    ▐   ░███   ░░███  ██████  ████████  ████   ██████   ████████    ▌
    ▐   ░███    ░███ ███░░███░░███░░███░░███  ░░░░░███ ░░███░░███   ▌
    ▐   ░███    ░███░███ ░███ ░███ ░░░  ░███   ███████  ░███ ░███   ▌
    ▐   ░███    ███ ░███ ░███ ░███      ░███  ███░░███  ░███ ░███   ▌
    ▐   ██████████  ░░██████  █████     █████░░████████ ████ █████  ▌
    ▐  ░░░░░░░░░░    ░░░░░░  ░░░░░     ░░░░░  ░░░░░░░░ ░░░░ ░░░░░   ▌
    ▐                                                               ▌
    ▐▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▌
    """
    print(logo)