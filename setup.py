from setuptools import setup, find_packages
from quool import __version__

with open('README.md', 'r') as f:
    ldes = f.read()

setup(
    name='quool',
    packages=find_packages(),
    author='ppoak',
    author_email='ppoak@foxmail.com',
    description='A Quantum Finance Analyze Toolkit',
    long_description=ldes,
    long_description_content_type='text/markdown',
    keywords=['Quantum', 'Finance'],
    url="https://github.com/ppoak/quool",
    version=__version__,
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'backtrader',
        'mplfinance',
        'rich',
    ],
    extras_require={
        "stats": [
            "statsmodules", 
            "sklearn", 
            "dask", 
            'scipy',
        ],
    },
    entry_points={
    }
)