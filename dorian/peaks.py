import numpy as np
import healpy as hp


def find_peaks(m, find_minima=False, fout=None):
    """Given a matrix m, returns an array with the values of the peaks found in it.
    A peak is defined as an element which is greater than its 8 neighbors.

    Parameters
    ----------
    m : 2D array
        Image where to look for peaks o minima.
    find_minima : bool
        Wheter to look for the minima instead of the peaks.
    fout : str
        Path of the folder which to save the output.

    Returns
    -------
    1D array
        Array containing the peaks/minima.
    """
    # Get shape of the matrix and prepare list for the peaks
    n, peaks = m.shape[0], []
    # Iterate over rows and columns (except for the borders)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            # Check that the element is greater/smaller than its 8 neighbors
            flag = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    if k == 0 and l == 0:
                        continue
                    if not (find_minima):
                        if m[i, j] <= m[i + k, j + l]:
                            flag = 1
                            break
                    else:
                        if m[i, j] >= m[i + k, j + l]:
                            flag = 1
                            break
                if flag == 1:
                    break
            if flag == 0:
                peaks.append(m[i, j])
    out = np.array(peaks)
    # Eventually save peaks/minima in numpy format
    if fout:
        if not (find_minima):
            fout1 = fout + "_peaks"
        else:
            fout1 = fout + "_minima"
        np.save(fout1, out)
        print(f"Saved extrema to {fout1}")
    return out


def find_peaks_fullsky(m, nest=False, find_minima=False, fout=None):
    """Given a HEALPix map, returns an array with the values of the peaks found
    in it. A peak is defined as an element which is greater than its 8 (or 7)
    HEALPix neighbors.

    Parameters
    ----------
    m : array-like
        HEALPix map where to look for peaks/minima.
    nest : bool
        Whether the angular map is nest ordered or not.
    find_minima : bool
        Wheter to look for the minima instead of the peaks.
    fout : str
        Path of the folder which to save the output.

    Returns
    -------
    1D array
        Array containing the indeces of the peaks/minima.
    1D array
        Array containing the values of the peaks/minima.
    """
    # Get size of the map and prepare list for the peaks and their locations
    npix, vals, locs = len(m), [], []
    nside = hp.npix2nside(npix)
    # Iterate over map
    for ipix in range(npix):
        # Check that the element is greater/smaller than its 8 neighbors
        flag = 0
        nbrs = hp.get_all_neighbours(nside, ipix, nest=nest)
        if -1 in nbrs:
            nbrs = np.delete(nbrs, np.where(nbrs == -1))
        for jpix in nbrs:
            if not (find_minima):
                if m[ipix] <= m[jpix]:
                    flag = 1
                    break
            else:
                if m[ipix] >= m[jpix]:
                    flag = 1
                    break
        if flag == 0:
            vals.append(m[ipix])
            locs.append(ipix)
    # Eventually save peaks/minima in numpy format
    if fout:
        out = np.stack((locs, vals), axis=-1)
        if not (find_minima):
            fout1 = fout + "_peaks"
        else:
            fout1 = fout + "_minima"
        np.save(fout1, out)
        print(f"Saved extrema to {fout1}")
    return locs, vals
