"""
Cylindrical TMAT patch 
"""
import jax.numpy as np
import numpy as onp
import jax
import treams.jcyl.cw as cw
import treams
from functools import partial

@partial(jax.jit, static_argnames=["mmax", "pol_mask"])
def globalt(mmax, kzs, k0, radius, positions, epsilons, pol_mask):
    num = positions.shape[1]

    tlocal = localt(mmax, kzs, k0, radius, num, epsilons)
    #tlocal = localt(mmax, kzs, k0, radii, epsilons)
    #jax.debug.print("tlocal: {}", tlocal)
    globalt, modes2, positions = globfromloc(
        tlocal, positions, mmax, kzs, k0, num, treams.Material(epsilons[-1]), pol_mask
    )

    return globalt, modes2, positions

def _cb_cylinder(mmax, kzs, k0, rad, epsilons):
    materials = [treams.Material(eps) for eps in epsilons]
    cyl = treams.TMatrixC.cylinder(kzs, mmax, k0.astype(complex), [rad], materials)
    cyl = cyl.changepoltype("parity")
    return np.array(cyl)

def cylinder(mmax, kzs, k0, rad, epsilons):
    size = (1+2*mmax)*2
    shape_dtype = jax.ShapeDtypeStruct((size, size), complex)
    return jax.pure_callback(_cb_cylinder, shape_dtype, mmax, kzs, k0, rad, epsilons)

@partial(jax.vmap, in_axes=(None, None, 0, None, None, None))
@partial(jax.jit, static_argnames=["mmax", "num"])
def localt(mmax, kzs, k0, radius, num, epsilons):
    tlocal = jax.scipy.linalg.block_diag(
        *[cylinder(mmax, kzs, k0, radius, epsilons)]*num
    )
    return tlocal

def filter_modes(modes, tmat, pol_filter):
    mask = modes[3] == pol_filter
    modes = np.array(modes)[:, mask]
    tmat = tmat[mask][:, mask]

    return modes, tmat

@partial(jax.vmap, in_axes=(0, None, None, None, 0, None, None, None))
def globfromloc(tlocal, positions, mmax, kzs, k0, num, material, pol_mask):
    modes = defaultmodes(mmax, kzs, num)
    positions = np.array(positions).T
    ind = positions[:, None, :] - positions
    rs = np.array(cw.car2cyl(*ind.T)).T
    rs = np.array(rs)
    # kn = k0 * material.n

    if pol_mask is not None:
        modes = np.array(modes)[:, pol_mask]
        tlocal = tlocal[pol_mask][:, pol_mask]
    pidx, kz, m, pol = modes
    krho = krhos(k0, kz, pol, material) #TODO diffable

    translation = cw.translate(
        *(m[:, None] for m in modes[1:]), #kz, m, pol
        *modes[1:],
        krho * rs[pidx[:, None], pidx, 0],
        rs[pidx[:, None], pidx, 1],
        rs[pidx[:, None], pidx, 2],
        singular=True,
    )
    
    translation = np.where(krho * rs[pidx[:, None], pidx, 0] != 0, translation, 0)

    B = tlocal @ np.reshape(translation, tlocal.shape)
    A = np.eye(tlocal.shape[0]) - B

    finalt = np.linalg.solve(
        A,
        tlocal,
    )
    return finalt, tuple(modes), positions

def refractive_index(epsilon, mu, kappa):
    epsilon = np.array(epsilon)
    n = np.sqrt(epsilon * mu)
    res = np.stack((n - kappa, n + kappa), axis=-1)
    res = res*np.where(np.imag(res) < 0, -1, 1)
    return res

def krhos(k0, kz, pol, material:treams.Material):
    ks = k0 * refractive_index(material.epsilon, material.mu, material.kappa)[pol]
    return np.where(kz == 0, ks, np.sqrt(ks * ks - kz * kz))


def defaultmodes(mmax, kzs, nmax=1):
    """
    Default sortation of modes

    Default sortation of the T-Matrix entries, including degree `l`, order `m` and
    polarization `p`.

    Args:
        lmax (int): Maximal value of `l`
        nmax (int, optional): Number of particles, defaults to `1`

    Returns:
        tuple
    """

    n = np.arange(0, nmax)
    kz = np.array(kzs)
    m = np.arange(-mmax, mmax+1)
    p = np.array([1, 0])
    modes = np.array(np.meshgrid(n, kz, m, p))
    return modes.reshape(4, -1)