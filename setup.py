#! /usr/bin/env python
"""Density Peak Advanced clustering algorithm, scikit-learn compatible."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('DPA', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'DPApipeline'
DESCRIPTION = 'The Density Peak Advanced packages.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
with open('requirements.txt') as f:
    INSTALL_REQUIRES = f.read().splitlines()
LICENSE = 'new BSD'
VERSION = __version__
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      description=DESCRIPTION,
      author="Maria d'Errico",
      license=LICENSE,
      version=VERSION,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(exclude=['notebooks']),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
