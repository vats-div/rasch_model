#!/usr/bin/env python

from setuptools import setup

setup(name='rasch_model',
      version='0.1',
      description='Rasch Model Learning',
      author='Divyanshu Vats',
      author_email='vats.div@gmail.com',
      url='https://github.com/vats-div/rasch_model',
      packages=['rasch_model'],
      install_requires=[
          'pandas', 'numpy', 'sklearn'
      ]
     )
