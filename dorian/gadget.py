import os, h5py
import numpy as np
from pathlib import Path
import healpy as hp
from .healpix import ud_grade_interp


def load_massmap(simDir, nmap):
    """Load the mass from the mass shell output of Gadget4.

    Parameters
    ----------
    simDir : str
        Path to the simulation output folder,
        e.g. '/virgo/simulations/MTNG/DM-Gadget4/MTNG-L500-270-A'.
    nmap : int
        Number of the mass-shell to load.

    Returns
    -------
    array-like
        HEALPix map containing the mass data.
    """
    mDir = f"{simDir}/mapsdir_{nmap:03d}/"
    offset = 0
    header = load_header(simDir, nmap)
    tot_pix, num_files = header["NpixTotal"], header["NumFiles"]
    mass = np.zeros(tot_pix, dtype=np.float64)
    for i in range(num_files):
        with h5py.File(f"{mDir}maps_{nmap:03d}.{i}.hdf5", "r") as f:
            m_temp = f["Maps"]["Mass"]
            # npix = header["NpixLocal"]
            # this should be the correct line, but sometimes NpixLocal is wrong
            npix = m_temp.shape[0]
            mass[offset : offset + npix] = m_temp
            offset += npix
    return mass


def load_header(simDir, nmap):
    """Load the header from the mass shell output of Gadget4.

    Parameters
    ----------
    simDir : str
        Path to the simulation output folder,
        e.g. '/virgo/simulations/MTNG/DM-Gadget4/MTNG-L500-270-A'.
    nmap : int
        Number of the mass-shell to load.

    Returns
    -------
    dict
        Dictionary containing the header. Keys are: Nmap, Redshift, AscaleStart,
        ComDistEnd, ComDistStart, NpixLocal, NpixTotal, Nside, NumFiles
    """
    with h5py.File(f"{simDir}/mapsdir_{nmap:03d}/maps_{nmap:03d}.0.hdf5", "r") as f:
        h = f["Header"]
        out = {}
        for k in h.attrs.keys():
            out[k] = h.attrs[k]
        out["Nmap"] = nmap
        out["Redshift"] = 1 / ((out["AscaleEnd"] + out["AscaleStart"]) * 0.5) - 1
    return out


def load_all_headers(simDir):
    """Load all the mass-shell info."""
    out = []
    for file in os.listdir(simDir):
        fname = os.fsdecode(file)
        if fname.startswith("mapsdir"):
            n = int(fname[-3:])
            out.append(load_header(simDir, nmap=n))
    out.sort(key=lambda x: x["Nmap"])
    return out


def load_params(simDir):
    """Load the parameters from the mass shell output of Gadget4.

    Parameters
    ----------
    simDir : str
        Path to the simulation output folder,
        e.g. '/virgo/simulations/MTNG/DM-Gadget4/MTNG-L500-270-A'.

    Returns
    -------
    dict
        Dictionary containing the parameters.
    """
    with h5py.File(f"{simDir}/mapsdir_000/maps_000.0.hdf5", "r") as f:
        h = f["Parameters"]
        params = {}
        for k in h.attrs.keys():
            params[k] = h.attrs[k]
    return params


def merge_massmaps(simDir, outDir, n_merge):
    """Merges every n mass-shells present in simDir, starting from the innermost one.

    Parameters
    ----------
    simDir : str
        Path to the simulation output folder.
    outDir : str
        Path where to save the new mass-shells.
    n_merge : int
        Number of the mass-shells to merge togehter.
    """
    print(f"Merging together every {n_merge} mass-shells form {simDir} to {outDir}")
    Path(f"{outDir}").mkdir(exist_ok=True)
    sh_tot = 0
    for f in os.listdir(simDir):
        if os.fsdecode(f).startswith("mapsdir"):
            sh_tot += 1
    h = load_header(simDir, 0)
    npix = h["NpixTotal"]
    mass_to_save, n_shells_cumulative = np.zeros(npix), 0
    ComDistEnd, AscaleEnd = 0, 1.0
    print(f"sh_tot: {sh_tot}")
    for n_shell in range(sh_tot - 1, -1, -1):
        n_shells_cumulative += 1
        h = load_header(simDir, n_shell)
        ComDistStart, AscaleStart = h["ComDistStart"], h["AscaleStart"]
        print(f"ComDistStart, AscaleStart: {ComDistStart, AscaleStart}")
        mass = load_massmap(simDir, n_shell)
        mass_to_save += mass
        if n_shells_cumulative % n_merge == 0:
            h = load_header(simDir, n_shell)
            ComDistStart, AscaleStart = h["ComDistStart"], h["AscaleStart"]
            print(f"Merging the above {n_merge} shells...")
            Path(f"{outDir}/mapsdir_{n_shell//n_merge:03d}").mkdir(exist_ok=True)
            with h5py.File(
                f"{simDir}/mapsdir_{n_shell:03d}/maps_{n_shell:03d}.0.hdf5", "r"
            ) as f_in:
                n_fac = n_shell // n_merge
                with h5py.File(
                    f"{outDir}/mapsdir_{n_fac:03d}/maps_{n_fac:03d}.0.hdf5", "w"
                ) as f_out:
                    for k1 in f_in.keys():
                        f_in.copy(k1, f_out)
                    del f_out["Maps/Mass"]
                    f_out["Maps"].create_dataset("Mass", data=mass_to_save, dtype="f4")
                    f_out["Header"].attrs["ComDistStart"] = ComDistStart
                    f_out["Header"].attrs["ComDistEnd"] = ComDistEnd
                    f_out["Header"].attrs["AscaleStart"] = AscaleStart
                    f_out["Header"].attrs["AscaleEnd"] = AscaleEnd
                    f_out["Header"].attrs["NpixLocal"] = f_out["Header"].attrs[
                        "NpixTotal"
                    ]
                    f_out["Header"].attrs["NumFiles"] = 1
            mass_to_save, n_shells_cumulative = np.zeros(npix), 0
            ComDistEnd, AscaleEnd = ComDistStart, AscaleStart
    print("Finished to merge the mass-shells, bye!")


def downsample_massmap(
    simDir,
    outDir,
    nside_out,
    n_start,
    n_end,
    nest=False,
    interpolation="bilinear",
    use_healpix_routine=False,
):
    """Downsamples the mass-shells present in simDir from n_start to n_end.

    Parameters
    ----------
    simDir : str
        Path to the simulation output folder.
    outDir : str
        Path where to save the new mass-shells.
    nside_out : int
        Healpix Nside of the downsampled mass-shells.
    interp : str
        Interpolation scheme to use, used only if use_healpix_routine is False
    use_healpix_routine : boolean
        If True, healpy ud_grade routine will be used
    """
    print(
        f"Downsampling to nside {nside_out} mass-shells from n.{n_start} to n.{n_end}.\n"
        + f"Simulation folder: {simDir}\n"
        + f"Output folder: {outDir}\n",
        flush=True,
    )

    Path(f"{outDir}").mkdir(exist_ok=True)
    for nmap in range(n_start, n_end + 1):
        mass = load_massmap(simDir, nmap)
        if use_healpix_routine:
            order = "NEST" if nest else "RING"
            mass_downsampled = hp.ud_grade(
                mass, nside_out, order_in=order, order_out=order, power=-2
            )
        else:
            mass_downsampled = ud_grade_interp(mass, nside_out, nest, interpolation)

        Path(f"{outDir}/mapsdir_{nmap:03d}").mkdir(exist_ok=True)

        with h5py.File(
            f"{simDir}/mapsdir_{nmap:03d}/maps_{nmap:03d}.0.hdf5", "r"
        ) as f_in:
            with h5py.File(
                f"{outDir}/mapsdir_{nmap:03d}/maps_{nmap:03d}.0.hdf5", "w"
            ) as f_out:
                for k1 in f_in.keys():
                    f_in.copy(k1, f_out)
                del f_out["Maps/Mass"]
                f_out["Maps"].create_dataset("Mass", data=mass_downsampled, dtype="f4")
                h_out = f_out["Header"]
                h_out.attrs["NpixLocal"] = hp.nside2npix(nside_out)
                h_out.attrs["NpixTotal"] = hp.nside2npix(nside_out)
                h_out.attrs["Nside"] = nside_out
                h_out.attrs["NumFiles"] = 1
        print(f"Done with massmap n. {nmap}", flush=True)

    print("Finished to downsample the mass-shells, bye!")



def write_massmap(
    outDir,
    nmap,
    mass_array,
    ComDistStart,
    ComDistEnd,
    Omega_M,
    Omega_L,
    HubbleParam,
):
    """Converts a HALPix map (given e.g. as a numpy array) in the Gadget4 massmap format,
    adopted by Dorian.

    Parameters
    ----------
    outDir : str
        Path where to save the massmaps.
    nmap : int
        Number of the massmap. These should be 0 for the outtermost shell and increasing
        as it reaches the observer (innermost shell)
    mass_array : array-like
        HEALPix map containing the binned mass, in units of 1e10*M_sun/h
    ComDistStart; float
        Outter boundary of the massmap, in Mpc/h
    ComDistEnd; float
        Inner boundary of the massmap, in Mpc/h
    Omega_M : float
        Matter density, cosmological parameter.
    Omega_L : float
        Cosmological constant density, cosmological parameter.
    HubbleParam : float
        Reduced Hubble constant, cosmological parameter.
    """
    Npix = len(mass_array)
    Path(f"{outDir}/mapsdir_{nmap:03d}").mkdir(exist_ok=True)
    with h5py.File(
        f"{outDir}/mapsdir_{nmap:03d}/maps_{nmap:03d}.0.hdf5", "w"
    ) as f_out:
        h_out = f_out.create_group("Header")
        h_out.attrs["ComDistStart"] = ComDistStart
        h_out.attrs["ComDistEnd"] = ComDistEnd
        h_out.attrs["NpixLocal"] = Npix
        h_out.attrs["NpixTotal"] = Npix
        h_out.attrs["Nside"] = hp.npix2nside(Npix)
        h_out.attrs["NumFiles"] = 1
        
        param_out = f_out.create_group("Parameters")
        param_out.attrs["Omega0"] = Omega_M
        param_out.attrs["OmegaLambda"] = Omega_L
        param_out.attrs["HubbleParam"] = HubbleParam
        
        f_out.create_group("Maps")
        f_out["Maps"].create_dataset("Mass", data=mass_array, dtype="f4")