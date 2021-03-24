import os
import numpy
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    if os.name == "nt":
        ext_comp_args = ['/openmp']
        ext_link_args = ['/openmp']
    else:
        ext_comp_args = ['-fopenmp']
        ext_link_args = ['-fopenmp']

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('CenterBasedClustering', parent_package, top_path)

    config.add_extension('CenterBasedClustering_',
                         sources=['CenterBasedClustering_.pyx'],
                         include_dirs=[numpy.get_include()],
                         define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                         language="c++",
                         extra_compile_args=ext_comp_args,
                         extra_link_args=ext_link_args,
                         libraries=libraries)

    config.ext_modules = cythonize(config.ext_modules, compiler_directives={'language_level': 3})

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
