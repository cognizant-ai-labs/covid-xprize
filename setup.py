"""
Code to allow this package to be pip-installed
"""

import os

import sys
from setuptools import setup, find_packages

LIBRARY_VERSION = '1.0.0'

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 6)

if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write('''
==========================
Unsupported Python version
==========================
This version of esp-sdk requires Python {}.{}, but you're trying to
install it on Python {}.{}.
'''.format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)

CUR_DIRECTORY_PATH = os.path.abspath(os.path.dirname(__file__))

# Python doesn't allow hyphens in package names so use underscore instead
PACKAGE_NAME = 'covid_xprize'
LIB_NAME = 'covid-xprize'


def read(fname):
    """
    Read file contents into a string
    :param fname: File to be read
    :return: String containing contents of file
    """
    with open(os.path.join(os.path.dirname(__file__), fname)) as file:
        return file.read()


setup(
    name=LIB_NAME,
    version=LIBRARY_VERSION,
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    packages=find_packages(),
    package_dir={PACKAGE_NAME: PACKAGE_NAME}, # the one line where all the magic happens
    package_data={
        PACKAGE_NAME: [
            'covid_xprize/examples/predictors/lstm/tests/fixtures/*',
            'covid_xprize/validation/data',
            'examples/predictors/lstm/data/*',
        ],
        '.': [
            'LICENSE.md'
        ]
    },
    install_requires=[
        'keras==2.4.3',
        'neat-python==0.92',
        'numpy==1.18.5',
        'pandas==1.1.2',
        'scikit-learn==0.23.2',
        'scipy==1.5.2',
        'setuptools==41.0.0',
        'tensorflow==2.2.1'
    ],
    description='Contains sample code and notebooks '
                'for developing and validating entries for the Cognizant COVID X-Prize.',
    long_description=read('README.md'),
    author='Olivier Francon, Darren Sargent, Elliot Meyerson',
    url='https://github.com/leaf-ai/covid-xprize/',
    license='See LICENSE.md'
)
