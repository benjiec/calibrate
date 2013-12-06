#!/usr/bin/env python

from setuptools import setup

setup(name='calibrate',
      version='0.9',
      description='Interpolate using a calibration curve',
      author='Benjie Chen',
      author_email='benjie@alum.mit.edu',
      packages=["."],
      package_dir={"": "."},
      install_requires=[
        'numpy',
        'scipy'
      ],
      classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Utilities'
      ],
     )

