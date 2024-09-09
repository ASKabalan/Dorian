import numpy as np
import healpy as hp


def parallel_transport_sphere(tv, v_theta, v_phi, w_theta, w_phi, tv_is_vector=False):
    """Parallel transports the tangent tensor (or vector) tv on the surface of a sphere
    along the geodesic connecting the position vectors v and w.
    (i.e. parallel transports tv from v to w)

    Parameters
    ----------
    tv : 2D-array or 2x2 matrix
        Tangent vector/tensor at position v to be parallel transported.
    v_theta : float
        Colatitude of initial position.
    v_phi : float
        Longitude of initial position.
    w_theta : float
        Colatitude of final position.
    w_phi : float
        Longitude of final position.
    tv_is_vector : bool
        Whether tv is a vector (i.e. 1st degree tensor).

    Returns
    -------
    float
        Parallel transported tangent tensor (vector) at position w.
    """
    # convert angles to 3D position vector
    v = hp.ang2vec(v_theta, v_phi)
    w = hp.ang2vec(w_theta, w_phi)

    # compute axis
    axis = np.cross(v, w)
    cosangle = np.dot(v, w)
    sinangle = np.linalg.norm(axis)
    axis = axis / sinangle if (sinangle != 0) else np.array([1.0, 0.0, 0.0])

    # define a vector which lays on the equator and is perpendicular to v
    p = np.array([-v[1], v[0], 0.0])
    # the following rotates p around axis by an angle defined by singangle and cosangle (rodrigues formula)
    rephi_v = (
        p * cosangle
        + np.cross(axis, p) * sinangle
        + axis * np.dot(axis, p) * (1 - cosangle)
    )

    # define ephi_w and etheta_w
    ephi_w = np.array([-w[1], w[0], 0.0])
    etheta_w = np.array([w[2] * w[0], w[2] * w[1], -1.0 * (w[0] ** 2 + w[1] ** 2)])

    # compute the final rotation angle
    norm = np.sqrt((1.0 - w[2]) * (1.0 + w[2]) * (1.0 - v[2]) * (1.0 + v[2]))
    if norm == 0.0:
        sinpsi = 0.0
        cospsi = 1.0
    else:
        sinpsi = (
            rephi_v[0] * etheta_w[0]
            + rephi_v[1] * etheta_w[1]
            + rephi_v[2] * etheta_w[2]
        ) / norm
        cospsi = (
            rephi_v[0] * ephi_w[0] + rephi_v[1] * ephi_w[1] + rephi_v[2] * ephi_w[2]
        ) / norm

    if tv_is_vector:
        # compute the parallel transported vector
        tw = np.zeros_like(tv)
        tw[0] = tv[0] * cospsi + tv[1] * sinpsi
        tw[1] = -1.0 * tv[0] * sinpsi + tv[1] * cospsi
    else:
        # compute the parallel transported tensor
        R = np.array([[cospsi, sinpsi], [-sinpsi, cospsi]])
        tw = R.dot(tv).dot(R.T)
    return tw


def parallel_transport_sphere_array(
    tv, v_theta, v_phi, w_theta, w_phi, tv_is_vector=False
):
    """Same as parallel_transport_sphere, but parallel transports an array of
    tensors (vectors) from the positions v to the positions w.
    In this case tv must be an array od N elements, and consequently v and w must be
    an array of N vectors.

    Parameters
    ----------
    tv : 1D array of 2D or 1D arrays (float)
        Tangent tensors (vectors) at position v to be parallel transported.
    v_theta : 1D array (float)
        Colatitudes of initial position.
    v_phi : 1D array (float)
        Longitudes of initial position.
    w_theta : 1D array (float)
        Colatitudes of final position.
    w_phi : 1D array (float)
        Longitudes of final position.
    tv_is_vector : bool
        Whether tv contains vectors (i.e. 1st degree tensor).

    Returns
    -------
    1D array of 2D or 1D arrays (float)
        Parallel transported tangent tensors (vectors) at positions w.
    """

    cospsi, sinpsi = get_rotation_angle_array(v_theta, v_phi, w_theta, w_phi)

    if tv_is_vector:
        # compute the parallel transported vectors
        tw = np.zeros_like(tv)
        tw[0] = tv[0] * cospsi + tv[1] * sinpsi
        tw[1] = -tv[0] * sinpsi + tv[1] * cospsi
    else:
        # compute the parallel transported tensors
        tw = rotate_tensor_array(tv, cospsi, sinpsi)
    return tw


############################# AUXYLIARY FUNCTIONS #############################


def get_rotation_angle_array_internal(v_theta, v_phi, w_theta, w_phi):
    N = v_theta.shape[0]

    # convert angles to 3D position vector
    vx = np.sin(v_theta)
    vz = np.cos(v_theta)
    w = hp.ang2vec(w_theta, w_phi - v_phi)
    w = np.ascontiguousarray(w.T)

    # compute axis
    axis = np.empty((3, N))
    axis[0] = -vz * w[1]
    axis[1] = vz * w[0] - vx * w[2]
    axis[2] = vx * w[1]

    cosangle = vx * w[0] + vz * w[2]
    sinangle = np.sqrt(
        axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2
    )  # np.linalg.norm(axis, axis=0)
    sin0 = np.where(sinangle != 0)
    axis.T[sin0] /= sinangle[sin0][:, np.newaxis]
    axis.T[np.where(sinangle == 0)] = np.array([1.0, 0.0, 0.0])

    # define a vector which lays on the equator and is perpendicular to v
    p = np.empty_like(w)
    p[1] = vx * cosangle
    p[0] = -axis[2] * vx * sinangle
    p[2] = axis[0] * vx * sinangle
    p += axis * (axis[1] * vx * (1 - cosangle))[np.newaxis, :]

    # define ephi_w and etheta_w
    etheta_w0 = w[2] * w[0]
    etheta_w1 = w[2] * w[1]
    etheta_w2 = -(w[0] ** 2 + w[1] ** 2)

    # compute the final rotation angle
    norm = np.sqrt((1.0 - w[2]) * (1.0 + w[2]) * (1.0 - vz) * (1.0 + vz))
    norm0 = np.where(norm == 0.0)
    sinpsi = (p[0] * etheta_w0 + p[1] * etheta_w1 + p[2] * etheta_w2) / norm
    sinpsi[norm0] = 0.0

    cospsi = (p[1] * w[0] - p[0] * w[1]) / norm
    cospsi[norm0] = 1.0
    return cospsi, sinpsi


def get_rotation_angle_array(v_theta, v_phi, w_theta, w_phi):
    """When parallel transporting a tensor of the sphere along a geodesic connecting the
    position v and w, it can be shown that with some strategic choices the whole operation
    can be reduced to a rotation of the tensor by an angle psi. This routine computes
    cosine and sine of such angle.

    Parameters
    ----------
    v_theta : float
        Colatitude of initial position.
    v_phi : float
        Longitude of initial position.
    w_theta : float
        Colatitude of final position.
    w_phi : float
        Longitude of final position.

    Returns
    -------
    1D array (float)
        Cosine of the rotation angles.
    1D array (float)
        Sine of the rotation angles.
    """
    N = v_theta.shape[0]
    bunchsize = 10000
    cospsi = np.empty(N)
    sinpsi = np.empty(N)
    lo = 0
    while lo < N:
        hi = min(N, lo + bunchsize)
        cospsi[lo:hi], sinpsi[lo:hi] = get_rotation_angle_array_internal(
            v_theta[lo:hi], v_phi[lo:hi], w_theta[lo:hi], w_phi[lo:hi]
        )
        lo = hi
    return cospsi, sinpsi


def rotate_tensor_array_internal(t, cospsi, sinpsi):
    msp = -sinpsi
    Rt00 = cospsi * t[0, 0] + sinpsi * t[1, 0]
    Rt01 = cospsi * t[0, 1] + sinpsi * t[1, 1]
    Rt10 = msp * t[0, 0] + cospsi * t[1, 0]
    Rt11 = msp * t[0, 1] + cospsi * t[1, 1]
    t[0, 0] = Rt00 * cospsi + Rt01 * sinpsi
    t[0, 1] = Rt00 * msp + Rt01 * cospsi
    t[1, 0] = Rt10 * cospsi + Rt11 * sinpsi
    t[1, 1] = Rt10 * msp + Rt11 * cospsi


def rotate_tensor_array(t, cospsi, sinpsi):
    """Rotates the tensors contained in t by the angles contained in psi
    via the rotation matrix R.

    R = |  cospsi  sinpsi |
        | -sinpsi  cospsi |

    t_r = R * t * R^-1

    Parameters
    ----------
    t : 1D array of 2D arrays (float)
        Tensors to be rotated.
    cospsi : 1D array (float)
        Cosine of the angles psi.
    sinpsi : 1D array (float)
        Sine of the angles psi.

    Returns
    -------
    1D array of 2D arrays (float)
        Parallel transported tangent tensors (vectors) at positions w.
    """
    N = t.shape[2]
    bunchsize = 5000
    lo = 0
    while lo < N:
        hi = min(N, lo + bunchsize)
        rotate_tensor_array_internal(t[:, :, lo:hi], cospsi[lo:hi], sinpsi[lo:hi])
        lo = hi
    return t
