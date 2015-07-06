#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, Extension, find_packages
from setuptools.dist import Distribution


with open('README.rst') as file:
    long_description = file.read()

setup(name='diffusions',
      version='1.0',
      description=('Affine Diffusions. Simulation and estimation.'),
      long_description=long_description,
      author='Stanislav Khrapov',
      license='MIT',
      author_email='khrapovs@gmail.com',
      url='https://github.com/khrapovs/diffusions',
      py_modules=['diffusions'],
      packages=find_packages(),
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
