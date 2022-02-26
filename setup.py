from setuptools import setup, find_packages

setup(
    name='ethash',
    version='0.1',
    py_modules=['ethash'],
    packages=find_packages(),
    install_requires=[
        'wheel',
        'pycryptodome==3.14.1',
        'numpy==1.22.2'
    ]
)