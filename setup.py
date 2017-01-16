from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

with open('requirements_dev.txt') as f:
    TEST_REQUIRES = [l.strip() for l in f.readlines() if l]

with open('README.md') as readme_file:
    readme = readme_file.read()

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

setup(name='easyML',
      version='0.1.0',
      description='A package designed to streamline the process of analyzing data using predictive models from scikit-learn\'s implementation.',
      long_description=readme,
      author='Aarshay Jain',
      license='BSD',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      author_email='aarshay.jain@columbia.edu',
      url='https://github.com/aarshayj/easyML',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      test_suite='tests',
      tests_require=TEST_REQUIRES
      )
