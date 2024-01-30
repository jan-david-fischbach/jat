"""
Cylindrical TMAT patch 
"""
import jax.numpy as np
import jax
import treams.jcyl.cw as cw
import treams


def globalt(mmax, kzs, k0, radii, positions, materials, pol_filter):
    num = radii.shape[0]
    tlocal = localt(mmax, kzs, k0, radii, materials)
    #jax.debug.print("tlocal: {}", tlocal)
    globalt, modes2, positions = globfromloc(
        tlocal, positions, mmax, kzs, k0, num, materials[-1], pol_filter=pol_filter
    )

    return globalt, modes2, positions

def cylinder(mmax, kzs, k0, rad, materials):
    #jax.debug.print("mmax: {}", mmax)
    #jax.debug.print("k0: {}", k0)
    #jax.debug.print("rad: {}", rad)
    cyl = treams.TMatrixC.cylinder(kzs, mmax, complex(k0), [rad], materials)
    cyl = cyl.changepoltype("parity")
    return np.array(cyl)


def localt(mmax, kzs, k0, radii, materials):
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

def globfromloc(tlocal, positions, mmax, kzs, k0, num, material, pol_filter=None):
    modes = defaultmodes(mmax, kzs, num)
    positions = np.array(positions).T
    ind = positions[:, None, :] - positions
    rs = np.array(cw.car2cyl(*ind.T)).T
    rs = np.array(rs)
    kn = k0 * material.n
    pidx, kz, m, pol = modes #pol not considered yet...
    if pol_filter is not None:
        mask = pol == pol_filter
        modes = np.array(modes)[:, mask]
        pidx, kz, m, pol = modes
        #jax.debug.print("tlocal.shape: {}", tlocal.shape)
        tlocal = tlocal[mask][:, mask]
        #jax.debug.print("tlocal.shape: {}", tlocal.shape)

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
    #jax.debug.print("translation: {}", translation)

    finalt = np.linalg.solve(
        np.eye(tlocal.shape[0]) - tlocal @ np.reshape(translation, tlocal.shape),
        tlocal,
    )
    return finalt, tuple(modes), positions

def krhos(k0, kz, pol, material):
    ks = k0 * material.nmp[pol]
    return np.sqrt(ks * ks - kz * kz)


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
        *np.array(
            [
                [n, kz, m, p]
                for n in range(0, nmax)
                for kz in kzs
                for m in range(-mmax, mmax+1)
                for p in range(1, -1, -1)
            ]
        ).T,
    )