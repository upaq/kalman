import sys
from setuptools import setup, find_packages

with open('requirements.txt', 'r') as req:
    requires = req.read().split('\n')

# enforce Python3 for all versions of pip/setuptools
assert sys.version_info >= (3,), 'This package requires Python 3.'

setup(name='kalman',
      version='0.1',
      description='Kalman filter for COVID-19',
      license='MIT',
      install_requires=requires,
      python_requires='>=3',
      packages=find_packages())
