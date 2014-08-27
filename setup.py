#!/usr/bin/env python

from setuptools import setup

setup(name='calibrate',
      version='0.9',
      description='Interpolate using a calibration curve',
      author='Benjie Chen',
      author_email='benjie@alum.mit.edu',
      packages=["calibrate"],
      package_dir={"calibrate": "."},
      install_requires=[
        'numpy',
        'scipy'
      ],
     )
