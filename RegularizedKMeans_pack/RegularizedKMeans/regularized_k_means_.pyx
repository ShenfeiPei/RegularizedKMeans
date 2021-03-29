# distutils: language = c++
cimport numpy as np
import numpy as np
np.import_array()

from .regularized_k_means_ cimport RegularizedKMeans

cdef class PyRegularizedKMeans:
    cdef RegularizedKMeans c_RegularizedKMeans
    def __cinit__(self, np.ndarray[double, ndim=2] X, int c_true, string init_method, bool warm_start, int n_jobs, unsigned int seed):
        self.c_RegularizedKMeans = RegularizedKMeans(X, c_true, init_method, warm_start, n_jobs, seed)

    def opt(self, int rep, string type):
        self.c_RegularizedKMeans.opt(rep, type)

    @property
    def y_pre(self):
        return np.array(self.c_RegularizedKMeans.Y)

    @property
    def obj(self):
        return self.c_RegularizedKMeans.obj