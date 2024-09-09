import numpy as np
from scipy import integrate
from scipy.optimize import root_scalar
from .constants import c_cgs


def E(z: float, 
      Omega_M: float, 
      Omega_L: float, 
      Omega_K: float = 0, 
      Omega_R: float = 0):
    """Computes the dimensionless Hubble parameter: E(z) = H/H_0

    Parameters
    ----------
    z : float
        Redshift.
    Omega_M : float
        Matter density, cosmological parameter.
    Omega_L : float
        Cosmological constant density, cosmological parameter.
    Omega_K : float
        Curvature density, cosmological parameter.
    Omega_R : float
        Radiation density, cosmological parameter.

    Returns
    -------
    float
        E(z).
    """
    z_1 = 1 + z
    E_2 = Omega_R * z_1**4 + Omega_M * z_1**3 + Omega_K * z_1**2 + Omega_L
    return np.sqrt(E_2)


def d_c(
    z: float, 
    Omega_M: float, 
    Omega_L: float, 
    Omega_K: float = 0, 
    Omega_R: float = 0
):
    """Computes the comoving distance (in Mpc/h) at the given redshift and the
    cosmological parameters.

    Parameters
    ----------
    z : float
        Redshift.
    Omega_M : float
        Matter density, cosmological parameter.
    Omega_L : float
        Cosmological constant density, cosmological parameter.
    Omega_K : float
        Curvature density, cosmological parameter.
    Omega_R : float
        Radiation density, cosmological parameter.

    Returns
    -------
    float
        Comoving distance (in Mpc/h).
    """
    E_z = lambda x: 1 / E(
        z=x,
        Omega_M=Omega_M,
        Omega_L=Omega_L,
        Omega_K=Omega_K,
        Omega_R=Omega_R,
    )
    dist = integrate.quad(E_z, 0, z)[0]
    dist *= c_cgs / 1e7
    return dist


def z_from_d_c(
    d_c_target: float,
    Omega_M: float,
    Omega_L: float,
    Omega_K: float = 0,
    Omega_R: float = 0,
    z_max: float = 1000,
):
    """Computes redshift given the comoving distance (in Mpc/h) and the
    cosmological parameters. It is the inverse function of d_c.

    Parameters
    ----------
    d_c_target : float
        Redshift.
    Omega_M : float
        Matter density, cosmological parameter.
    Omega_L : float
        Cosmological constant density, cosmological parameter.
    Omega_K : float
        Curvature density, cosmological parameter.
    Omega_R : float
        Radiation density, cosmological parameter.
    z_max : float
        Upper limit to the redshift range, i.e. the value will be
        looked for in the range [0, z_max].

    Returns
    -------
    float
        Redshift.
    """
    f = lambda z: d_c(z, Omega_M, Omega_L, Omega_K, Omega_R) - d_c_target
    return root_scalar(f, bracket=[0.0, z_max]).root
