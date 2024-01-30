import jax.numpy as jnp
import scipy.special
from jax import custom_jvp, pure_callback
import jax
# see https://github.com/google/jax/issues/11002


def generate_bessel(function):
    """function is Jv, Yv, Hv_1,Hv_2"""

    @custom_jvp
    def cv(v, x):
        return pure_callback(
            lambda vx: function(*vx),
            x,
            (v, x),
            vectorized=True,
        )

    @cv.defjvp
    def cv_jvp(primals, tangents):
        v, x = primals
        dv, dx = tangents
        primal_out = cv(v, x)

        # https://dlmf.nist.gov/10.6 formula 10.6.1
        tangents_out = jax.lax.select(
            v == 0,
            -cv(v + 1, x),
            0.5 * (cv(v - 1, x) - cv(v + 1, x)),
        )

        #jax.debug.print("tangents_out {}", tangents_out)
        #jax.debug.print("primal_out {}", primal_out)
        #jax.debug.print("v {}", v)
        #jax.debug.print("x {}", x)

        tangents_out = jnp.where(jnp.abs(x)<1e-30, 0, tangents_out*dx)

        jax.debug.print("dx")

        return primal_out, tangents_out

    return cv


jv = generate_bessel(scipy.special.jv)
yv = generate_bessel(scipy.special.yv)
hankel1 = generate_bessel(scipy.special.hankel1)
hankel2 = generate_bessel(scipy.special.hankel2)


# def generate_modified_bessel(function, sign):
#     """function is Kv and Iv"""

#     @custom_jvp
#     def cv(v, x):
#         return pure_callback(
#             lambda vx: function(*vx),
#             x,
#             (v, x),
#             vectorized=True,
#         )

#     @cv.defjvp
#     def cv_jvp(primals, tangents):
#         v, x = primals
#         dv, dx = tangents
#         primal_out = cv(v, x)

#         # https://dlmf.nist.gov/10.6 formula 10.6.1
#         tangents_out = jax.lax.cond(
#             v == 0,
#             lambda: sign * cv(v + 1, x),
#             lambda: 0.5 * (cv(v - 1, x) + cv(v + 1, x)),
#         )

#         return primal_out, tangents_out * dx

#     return cv


# kv = generate_modified_bessel(scipy.special.kv, sign=-1)
# iv = generate_modified_bessel(scipy.special.iv, sign=+1)


def spherical_bessel_genearator(f):
    def g(v, x):
        return f(v + 0.5, x) * jnp.sqrt(jnp.pi / (2 * x))

    return g


spherical_jv = spherical_bessel_genearator(jv)
spherical_yv = spherical_bessel_genearator(yv)
spherical_hankel1 = spherical_bessel_genearator(hankel1)
spherical_hankel2 = spherical_bessel_genearator(hankel2)