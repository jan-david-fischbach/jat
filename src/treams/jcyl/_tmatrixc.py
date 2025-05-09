"""
Cylindrical TMAT patch 
"""
import jax.numpy as np
import numpy as onp
import jax
import treams.jcyl.cw as cw
import treams
from functools import partial


def globalt(mmax, kzs, k0, radii, positions, materials, pol_mask):
    num = radii.shape[0]

    size = num*(1+2*mmax)*2

    epsilons = [mat.epsilon for mat in materials]
    shape_dtype = jax.ShapeDtypeStruct((size, size), complex)
    tlocal = jax.pure_callback(localt, shape_dtype, mmax, kzs, k0, radii, epsilons)
    #tlocal = localt(mmax, kzs, k0, radii, epsilons)
    #jax.debug.print("tlocal: {}", tlocal)
    globalt, modes2, positions = globfromloc(
        tlocal, positions, mmax, kzs, k0, num, materials[-1], pol_mask=pol_mask
    )

    return globalt, modes2, positions

def cylinder(mmax, kzs, k0, rad, materials):
    #jax.debug.print("mmax: {}", mmax)
    #jax.debug.print("k0: {}", k0)
    #jax.debug.print("rad: {}", rad)
    cyl = treams.TMatrixC.cylinder(kzs, mmax, k0.astype(complex), [rad], materials)
    cyl = cyl.changepoltype("parity")
    return np.array(cyl)


def localt(mmax, kzs, k0, radii, epsilons):
    materials = [treams.Material(eps) for eps in epsilons]
    num = radii.shape[0]
    mycyl = cylinder(mmax, kzs, k0, radii[0], materials)
    shape = mycyl.shape[0]

    for i in range(1, num):
        nextcyl= cylinder(mmax, kzs, k0, radii[i], materials)
        mycyl = np.concatenate(
            (mycyl, np.zeros((shape, num * shape)), nextcyl), axis=1
        )

    tlocal = np.vstack(tuple(np.split(mycyl, num, axis=1)))
    return tlocal

def filter_modes(modes, tmat, pol_filter):
    mask = modes[3] == pol_filter
    modes = np.array(modes)[:, mask]
    tmat = tmat[mask][:, mask]

    return modes, tmat

def globfromloc(tlocal, positions, mmax, kzs, k0, num, material, pol_mask=None):
    modes = defaultmodes(mmax, kzs, num)
    positions = np.array(positions).T
    ind = positions[:, None, :] - positions
    rs = np.array(cw.car2cyl(*ind.T)).T
    rs = np.array(rs)
    # kn = k0 * material.n

    if pol_mask is not None:
        modes = onp.array(modes)[:, pol_mask]
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

def krhos(k0, kz, pol, material:treams.Material):
    ks = k0 * material.nmp[pol]
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
    return (
        *onp.array(
            [
                [n, kz, m, p]
                for n in range(0, nmax)
                for kz in kzs
                for m in range(-mmax, mmax+1)
                for p in range(1, -1, -1)
            ]
        ).T,
    )