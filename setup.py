try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

NAME = "pysa"

config = {
    'description': 'Signal analysis toolbox for Hilbert Huang Transform',
    'author': 'Joar Molvaer, Fredrik Worren',
    'url': 'https://github.com/FWorren/pysa',
    'download_url': 'https://github.com/FWorren/pysa',
    'author_email': 'freworr@gmail.com',
    'version': '1.0',
    'install_requires': ['nose', 'scipy', 'numpy', 'matplotlib'],
    'packages': [NAME],
    'scripts': [],
    'name': NAME
}

setup(**config)
