#ifndef CENTER_BASED_CLUSTERING_H_
#define CENTER_BASED_CLUSTERING_H_

#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <utility>

using namespace std;

class CenterBasedClustering {

public:
    int num;
    int dim;
    int c_true;

    unsigned int seed;
    string init_method;

    vector<vector<double>> X;
    vector<vector<double>> Cen;
    vector<int> y;

    vector<double> xnorm;
    vector<double> cen_norm;

    vector<vector<double>> obj;
    vector<vector<int>> Y;

    vector<int> n;

    CenterBasedClustering();
    CenterBasedClustering(vector<vector<double>>& X, int c_true, string init_method, unsigned int seed);
    ~CenterBasedClustering();
    void Matrix_Fnorm_byRow(vector<vector<double>> &M, vector<double> &mnorm);

    double GetSumSquaredError();
    void Update_Cen();

    void initial_y();
    void InitWithRandomAssignment();

    void Update_n();

private:
    default_random_engine el_;
    uniform_int_distribution<int> uniform_rand;
};

#endif  // CENTER_BASED_CLUSTERING_H_
