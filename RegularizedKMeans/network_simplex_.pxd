from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "network_simplex.cpp":
    pass

cdef extern from "network_simplex.h":
    cdef cppclass NetworkSimplex:

        NetworkSimplex() except +
        void BuildHard(const vector[vector[double]] &costs, int k, int lower_bound, int upper_bound)
        void Build(const vector[vector[double]] &costs, int f_th)
        void Simplex()
        void UpdateCosts(vector[vector[double]]& costs)
        void GetAssignments(vector[int] & assignments) const
        double min_cost() const
        double regularized_f(int i, double a, double b)