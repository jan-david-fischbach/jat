from functools import partial
import jax.numpy as jnp
import numpy as np
import scipy.special
from jax import custom_jvp, pure_callback
import jax
# see https://github.com/google/jax/issues/11002

def make_maskable(function):
    def maskable(v,x,mask):
        masked_v = np.array(v)[mask]
        masked_x = np.array(x)[mask]
        result = np.zeros_like(x)
        result[mask] = function(masked_v,masked_x)
        return result
    return maskable

def generate_bessel(function):
    """function is Jv, Yv, Hv_1,Hv_2"""

    function = make_maskable(function)

    @custom_jvp
    def cv(v, x, mask=True):
        mask2 = x==0
        res = pure_callback(
                lambda vx: function(*vx),
                x,
                (v, x, mask),
                vectorized=True,
            )

        return jnp.where(mask2, 0, res)

    @cv.defjvp
    def cv_jvp(primals, tangents):
        v, x, mask = primals
        dv, dx = tangents
        primal_out = cv(v, x)

        # https://dlmf.nist.gov/10.6 formula 10.6.1
        tangents_out = jax.lax.select(
            v == 0,
            -cv(v + 1, x, mask),
            0.5 * (cv(v - 1, x, mask) - cv(v + 1, x, mask)),
        )

        tangents_out = jnp.where(jnp.abs(x)<1e-30, 0, tangents_out*dx)

        return primal_out, tangents_out

    return cv


jv = generate_bessel(scipy.special.jv)
yv = generate_bessel(scipy.special.yv)
hankel1 = generate_bessel(scipy.special.hankel1)
hankel2 = generate_bessel(scipy.special.hankel2)

def spherical_bessel_generator(f):
    def g(v, x):
        return f(v + 0.5, x) * jnp.sqrt(jnp.pi / (2 * x))

    return g


spherical_jv = spherical_bessel_generator(jv)
spherical_yv = spherical_bessel_generator(yv)
spherical_hankel1 = spherical_bessel_generator(hankel1)
spherical_hankel2 = spherical_bessel_generator(hankel2)