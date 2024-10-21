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

cdef float BRANCH_CUT_INCGAMMA = 0.5*pi
cdef float BRANCH_CUT_CPOW     = -0.5*pi

cdef float BRANCH_CUT_SQRT_MIE_N = -pi
cdef float BRANCH_CUT_SQRT_MIE_Z = -pi
#rotated from the typical negative real axis

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