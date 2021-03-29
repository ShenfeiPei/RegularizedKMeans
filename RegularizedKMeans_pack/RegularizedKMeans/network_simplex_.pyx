# distutils: language = c++
cimport numpy as np
import numpy as np
np.import_array()

from .network_simplex_ cimport NetworkSimplex

cdef class PyNetworkSimplex:
    cdef NetworkSimplex c_NetworkSimplex

    def __cinit__(self):
        self.c_NetworkSimplex = NetworkSimplex()

    def BuildHard(self, np.ndarray[double, ndim=2] costs, int k, int lower_bound, int upper_bound):
        self.c_NetworkSimplex.BuildHard(costs, k, lower_bound, upper_bound)

    def Build(self, np.ndarray[double, ndim=2] costs, int f_th):
        self.c_NetworkSimplex.Build(costs, f_th)

    def Simplex(self):
        self.c_NetworkSimplex.Simplex()

    def UpdateCosts(self, np.ndarray[double, ndim=2] costs):
        self.c_NetworkSimplex.UpdateCosts(costs)

    def GetAssignments(self, np.ndarray[int, ndim=1] assignments):
        self.c_NetworkSimplex.GetAssignments(assignments)

    def min_cost(self):
        return self.c_NetworkSimplex.min_cost()

    def regularized_f(self, int i, double a, double b):
        return self.c_NetworkSimplex.regularized_f(i, a, b)