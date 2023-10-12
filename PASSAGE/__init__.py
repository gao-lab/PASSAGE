'''
PASSAGE (Phenotype Associated Spatial Signature Analysis with Graph-based Embedding)
'''

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution

    def version(name):
        return get_distribution(name).version


from . import model
from . import viz

name = "PASSAGE"
__version__ = version(name)
__author__ = 'Chen-Kai Guo'