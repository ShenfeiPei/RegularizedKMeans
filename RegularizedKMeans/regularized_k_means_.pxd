from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from ..CenterBasedClustering.CenterBasedClustering_ cimport CenterBasedClustering
from .network_simplex_ cimport NetworkSimplex

cdef extern from "regularized_k_means.cpp":
    pass

cdef extern from "regularized_k_means.h":
    cdef cppclass RegularizedKMeans(CenterBasedClustering):

        vector[double] obj
        RegularizedKMeans() except +
        RegularizedKMeans(vector[vector[double]] &data, int c_true, string init_method, bool warm_start, int n_jobs, unsigned int seed) except +

        double SolveHard()
        double SolveHard(int lower_bound, int upper_bound)
        double Solve(int f_th)
        void opt(int rep, string type)