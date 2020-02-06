#!/usr/bin/env python
from __future__ import print_function
"""CasingSimulations: Numerical simulations of electromagnetic surveys over
in settings where steel cased wells are present.
"""

from distutils.core import setup
from setuptools import find_packages

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Natural Language :: English',
]

with open("README.md") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())

setup(
    name="simpegskytem",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.7',
        'scipy>=0.13',
        'cython',
        'pymatsolver>=0.1.1',
        'ipython',
        'ipywidgets',
        'jupyter',
        'matplotlib',
        'properties[math]',
        'SimPEG',
    ],

    author="Seogi Kang",
    author_email="sgkang09@gmail.com",
    description="3D AEM simulation",
    long_description=LONG_DESCRIPTION,
    license="MIT",
    keywords="geophysics electromagnetics",
    url="http://github.com/lheagy/casingResearch",
    download_url="https://github.com/simpeg-research/kang-2019-3D-aem",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3=False
)
