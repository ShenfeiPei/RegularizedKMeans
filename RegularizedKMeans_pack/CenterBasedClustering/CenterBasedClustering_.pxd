from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "CenterBasedClustering.cpp":
    pass

cdef extern from "CenterBasedClustering.h":
    cdef cppclass CenterBasedClustering:
        int num
        int dim
        int c_true

        unsigned int seed
        string init_method

        vector[vector[double]] X
        vector[vector[double]] Cen
        vector[int] y

        vector[double] xnorm
        vector[double] cen_norm

        vector[vector[double]] obj;
        vector[vector[int]] Y;

        vector[int] n;

        CenterBasedClustering() except +
        CenterBasedClustering(vector[vector[double]] &X, int c_true, string init_method, unsigned int seed) except +
        void Matrix_Fnorm_byRow(vector[vector[double]] &M, vector[double] &mnorm)

        double GetSumSquaredError()
        void Update_Cen()

        void initial_y()
        void InitWithRandomAssignment()

        void Update_n()
