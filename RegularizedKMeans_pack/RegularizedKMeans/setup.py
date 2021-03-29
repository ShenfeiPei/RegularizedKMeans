import os
import numpy
from Cython.Build import cythonize
from RegularizedKMeans_pack.Public import cg


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('RegularizedKMeans', parent_package, top_path)

    config.add_extension('network_simplex_',
                         sources=['network_simplex_.pyx'],
                         include_dirs=[numpy.get_include()],
                         define_macros=cg.define_macros,
                         language="c++",
                         extra_compile_args=cg.ext_comp_args,
                         extra_link_args=cg.ext_link_args,
                         libraries=cg.libraries)

    config.add_extension('regularized_k_means_',
                         sources=['regularized_k_means_.pyx'],
                         include_dirs=[numpy.get_include()],
                         define_macros=cg.define_macros,
                         language="c++",
                         extra_compile_args=cg.ext_comp_args,
                         extra_link_args=cg.ext_link_args,
                         libraries=cg.libraries)

    config.ext_modules = cythonize(config.ext_modules, compiler_directives={'language_level': 3})

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
