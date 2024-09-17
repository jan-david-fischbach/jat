import jax.numpy as np
import treams.jspecial as jts


def translate(
    kz, mu, pol, qz, m, qol, krr, phi, z, singular=True,
):
    if singular:
        return translate_s(kz, mu, pol, qz, m, qol, krr, phi, z)
    return translate_r(kz, mu, pol, qz, m, qol, krr, phi, z)

def translate_s(kz, mu, pol, qz, m, qol, krr, phi, z):
    mask = (pol == qol) #& ((pol == 0) | (pol == 1))
    answer = tl_vcw(kz, mu, qz, m, krr, phi, z, mask) * mask
    return answer

def translate_r(kz, mu, pol, qz, m, qol, krr, phi, z):
    mask = (pol == qol) #& ((pol == 0) | (pol == 1))
    answer = tl_vcw_r(kz, mu, qz, m, krr, phi, z, mask) * mask 
    return answer

def tl_vcw(kz, mu, qz, m, krr, phi, z, mask): #singular
    mask = np.logical_and(mask, kz == qz)
    return jts.hankel1(m - mu, krr) * np.exp(1j * ((m - mu) * phi + kz * z)) * mask


def tl_vcw_r(kz, mu, qz, m, krr, phi, z, mask): #regular
    mask = np.logical_and(mask, kz == qz)
    return jts.jv(m - mu, krr) * np.exp(1j * ((m - mu) * phi + kz * z)) * mask

def car2cyl(x,y,z):
    mask = np.logical_and(x==0, y==0)
    x = np.where(mask, 1, x)
    phi = np.arctan2(y,x)
    phi = np.where(mask, 0, phi)
    rho = np.hypot(x,y)
    rho = np.where(mask, 0, rho)
    return rho, phi, z
    