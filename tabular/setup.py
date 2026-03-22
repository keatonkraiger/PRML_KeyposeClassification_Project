from setuptools import setup, find_packages

# Read the contents of requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='tabular',
    version='0.01',
    packages=find_packages(),
    install_requires=requirements
)
