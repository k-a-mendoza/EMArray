from setuptools import setup, find_packages
setup(
    name='emarray',
    version='1.0.0b',
    packages=find_packages(exclude=['tests*','examples and tutorials*']),
    license='none',
    description='core mt tooling for working with arrays of transfer functions',
    long_description=open('readme.md').read(),
    install_requires=[],
    author='Kevin A Mendoza',
    author_email='kevinmendoza@icloud.com')