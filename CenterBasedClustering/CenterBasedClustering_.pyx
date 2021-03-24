# distutils: language = c++
cimport numpy as np
import numpy as np
np.import_array()

from .CenterBasedClustering_ cimport CenterBasedClustering

cdef class PyCenterBasedClustering:
    cdef CenterBasedClustering c_CenterBasedClustering

    def __cinit__(self, np.ndarray[double, ndim=2] X, int c_true, string init_method, unsigned int seed):
        self.c_CenterBasedClustering = CenterBasedClustering(X, c_true, init_method, seed)

    def Matrix_Fnorm_byRow(self, np.ndarray[double, ndim=2] M, np.ndarray[double, ndim=1] mnorm):
        self.c_CenterBasedClustering.Matrix_Fnorm_byRow(M, mnorm)

    def GetSumSquaredError(self):
        return self.c_CenterBasedClustering.GetSumSquaredError()

    def Update_Cen(self):
        self.c_CenterBasedClustering.Update_Cen()

    def initial_y(self):
        self.c_CenterBasedClustering.initial_y()

    def InitWithRandomAssignment(self):
        self.c_CenterBasedClustering.InitWithRandomAssignment()

    def Update_n(self):
        self.c_CenterBasedClustering.Update_n()



