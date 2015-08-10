#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy

with open('README.rst') as file:
    long_description = file.read()

kwargs = {'libraries': [], 'include_dirs': [numpy.get_include()]}

ext_modules = [Extension('diffusions.simulate',
                         ['./diffusions/simulate.pyx'], **kwargs)]

setup(name='diffusions',
      version='1.0',
      description='Affine Diffusions. Simulation and estimation.',
      long_description=long_description,
      author='Stanislav Khrapov',
      author_email='khrapovs@gmail.com',
      url='https://github.com/khrapovs/diffusions',
      license='MIT',
      packages=find_packages(),
      ext_modules=ext_modules,
      cmdclass={'build_ext': build_ext},
      zip_safe=False,
      keywords=['diffusion', 'econometrics', 'estimation', 'affine',
                'CIR', 'Vasicek', 'Brownian motion', 'SDE', 'GBM',
                'Heston', 'volatility', 'stochastic', 'central tendency',
                'GMM'],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
      ],
      )
