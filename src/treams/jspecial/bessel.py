import functools
import jax
import jax.numpy as jnp
import scipy.special as scs


def make_bessel_diffable(f: callable) -> callable:
    @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
    def _f(n, z):
        return f(n, z)

    @_f.defjvp
    def _f_jvp(primals, tangents):
        n, z = primals
        n_dot, z_dot = tangents

        primal_out = _f(n, z)
        if n == 0:
            df_dz = -_f(1, z)
        else:
            df_dz = _f(n - 1, z) - (n + 1) / z * primal_out
        return primal_out, df_dz*z_dot

    def scipy_signature(n: int, z: complex, derivative: bool = False) -> complex:
        if not derivative:
            return _f(n, z)
        d_f = jax.grad(_f, argnums=1)
        return d_f(n, z)

    return scipy_signature

#spherical_jn = make_bessel_diffable(scs.spherical_jn)
spherical_yn = make_bessel_diffable(scs.spherical_yn)

def spherical_hankel1(n: int, z: complex) -> complex:
    return scs.spherical_jn(n, z) + 1j*scs.spherical_yn(n, z)

def spherical_hankel2(n: int, z: complex) -> complex:
    return scs.spherical_jn(n, z) - 1j*scs.spherical_yn(n, z)

spherical_h1 = make_bessel_diffable(spherical_hankel1)
spherical_h2 = make_bessel_diffable(spherical_hankel2)

bessel_jn = jax.scipy.special.bessel_jn

@functools.partial(jax.jit, static_argnames=["v"])
def spherical_jn(z, v: int):
    return jnp.sqrt(jnp.pi/(2*z)) * bessel_jn(z, v=v+0.5)