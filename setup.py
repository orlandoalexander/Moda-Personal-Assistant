from setuptools import setup
from setuptools import find_packages

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='preproc',
      description="preprocess images and convert to numpy arrays",
      packages=find_packages(), # find packages automatically
      install_requires=requirements,  # install dependencies when install package
      package_data={'preproc_data': ['preproc_data/*']}) # load package data
