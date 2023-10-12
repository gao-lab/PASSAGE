from setuptools import Command, find_packages, setup

__lib_name__ = "PASSAGE"
__lib_version__ = "1.0.0"
__description__ = "Learning phenotype associated signature in spatial transcriptomics with PASSAGE"
__url__ = "https://github.com/gao-lab/PASSAGE-dev"
__author__ = "Chenkai Guo"
__author_email__ = "guo_chenkai@gibh.ac.cn"
__license__ = "MIT"
__keywords__ = ["Spatial transcriptomics", "spatial signature identification", "deep learning", "graph neural network"]
__requires__ = ["requests",]

# with open("README.rst", "r", encoding="utf-8") as f:
#     __long_description__ = f.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['PASSAGE'],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    long_description = None
)
