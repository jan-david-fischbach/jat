"""Configuration.

This is the place to define global configuration variables. Use, e.g.,

.. code-block::

   import treams


   treams.config.POLTYPE = 'parity'
   # The rest of the code

to change the default value globally. Currently the only global configuration variable
is ``POLTYPE`` which defines the default polarization type. It can be either `helicity`
or `parity`.
"""
from libc.math cimport pi

POLTYPE = "helicity"
BRANCH_CUT_SQRT_KZ = 0.5*pi

cdef float BRANCH_CUT_INCGAMMA = 0.5*pi
cdef float BRANCH_CUT_CPOW     = -0.5*pi
# rotated from the typical negative real axis

cdef float BRANCH_CUT_SQRT_MIE_N = -pi
cdef float BRANCH_CUT_SQRT_MIE_Z = -pi

cdef float SINGULARITY_REDINCGAMMA = 1e-7
cdef float SINGULARITY_THRESH_REDINCGAMMA = 1e-12 
# Singularity clipping is triggered later for smaller values of thresh

def set_BRANCH_CUT_INCGAMMA(val):
   global BRANCH_CUT_INCGAMMA
   BRANCH_CUT_INCGAMMA = val

def set_BRANCH_CUT_CPOW(val):
   global BRANCH_CUT_CPOW
   BRANCH_CUT_CPOW = val

def set_BRANCH_CUT_SQRT_MIE_N(val):
   global BRANCH_CUT_SQRT_MIE_N
   BRANCH_CUT_SQRT_MIE_N = val

def set_BRANCH_CUT_SQRT_MIE_Z(val):
   global BRANCH_CUT_SQRT_MIE_Z
   BRANCH_CUT_SQRT_MIE_Z = val

def set_BRANCH_CUT_SQRT_KZ(val):
   global BRANCH_CUT_SQRT_KZ
   BRANCH_CUT_SQRT_KZ = val


def set_SINGULARITY_REDINCGAMMA(val):
   global SINGULARITY_REDINCGAMMA
   SINGULARITY_REDINCGAMMA = val

def set_SINGULARITY_THRESH_REDINCGAMMA(val):
   global SINGULARITY_THRESH_REDINCGAMMA
   SINGULARITY_THRESH_REDINCGAMMA = val