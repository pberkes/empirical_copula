#!/usr/bin/env python

from setuptools import setup


setup(
    name='empirical_copula',
    version='1.0',
    description='Empirical copulas for discrete variables',
    author='Pietro Berkes',
    author_email='pietro.berkes@gmail.com',
    url='https://github.com/pberkes/empirical_copula',
    packages=['empirical_copula'],
    install_requires=['numpy', 'pandas']
)
