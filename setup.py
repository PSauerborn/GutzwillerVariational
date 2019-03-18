from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='GutzwillerOpt', ext_modules=cythonize('gutzwiller_functions.pyx'), include_dirs=[np.get_include()])
