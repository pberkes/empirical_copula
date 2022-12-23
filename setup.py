#!/usr/bin/env python

from setuptools import setup

from empirical_copula import __version__


setup(
    name='empirical_copula',
    version=str(__version__),
    description='Empirical copulas for discrete variables',
    author='Pietro Berkes',
    author_email='pietro.berkes@gmail.com',
    url='https://github.com/pberkes/empirical_copula',
    packages=['empirical_copula'],
    install_requires=['numpy', 'pandas']
)
