#!/usr/bin/env python
from setuptools import setup
import os
import pyageng

setup(name='pyageng',
      version = pyageng.__version__,
      description='Code for the book "Measurements and Data Analysis for Agricultural Engineers using Python"',
      author='Matti Pastell',
      author_email='matti.pastell@helsinki.fi',
      url='http://pyageng.mpastell.com',
      packages=['pyageng'],
      license='LICENSE.txt',
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4'
        ]
)
