import quool
from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "quool",
    packages = ["quool"],
    author = "ppoak",
    author_email = "ppoak@foxmail.com",
    description = "Quantitative Toolkit - a helper in quant developping",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    keywords = ['quant', 'framework', 'finance'],
    url = "https://github.com/ppoak/quool",
    version = quool.__version__,
    install_requires = [
        'bs4',
        'tqdm',
        'lxml',
        'numpy',
        'joblib',
        'pandas',
        'pyarrow',
        'retrying',
        'requests',
        'matplotlib',
        'backtrader',
    ],
)