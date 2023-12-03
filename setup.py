from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "quool",
    packages = find_packages(),
    author = "ppoak",
    author_email = "ppoak@foxmail.com",
    description = "Quantitative Toolkit - a helper in quant developping",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    keywords = ['quant', 'framework', 'finance'],
    url = "https://github.com/ppoak/quool",
    version = '0.1.0',
    install_requires = [
        'bs4',
        'numpy',
        'joblib',
        'pandas',
        'matplotlib',
    ],
    extras_require={
        'backtest': ['backtrader'],
        'crawler': ['akshare'],
        'all': ['backtrader', 'akshare'],
    }
)