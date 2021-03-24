#include "CenterBasedClustering.h"

CenterBasedClustering::CenterBasedClustering(){}

CenterBasedClustering::CenterBasedClustering(vector<vector<double>> &X, int c_true, string init_method, unsigned int seed){
    this->num = X.size();
    this->dim = X[0].size();
    this->c_true = c_true;

    this->seed = seed;
    this->init_method = std::move(init_method);
    el_.seed(seed);
    uniform_rand = uniform_int_distribution<int>(0, this->c_true-1);

    // allocation
    this->X = X;
    this->xnorm = vector<double>(num, 0);

    this->Cen = vector<vector<double>>(this->c_true, vector<double>(this->dim, 0));
    this->cen_norm = vector<double>(this->c_true, 0);

    this->y = vector<int>(this->num, 0);

    this->n = vector<int>(this->c_true, 0);
}

CenterBasedClustering::~CenterBasedClustering(){}

/// mnorm[i] = sum(M[i, :]**2)
/// \param M: vector<vector<double>> &
/// \param mnorm: vector<double> &
void CenterBasedClustering::Matrix_Fnorm_byRow(vector<vector<double>> &M, vector<double> &mnorm){
    #pragma omp parallel for
    for (int i = 0; i < M.size(); i++) {
        mnorm[i] = inner_product(M[i].begin(), M[i].end(), M[i].begin(), (double) 0);
    }
}

/// n[i] = sum(y == i)
void CenterBasedClustering::Update_n(){
    fill(n.begin(), n.end(), 0);
    int tmp_c = 0;
    for (int i = 0; i < num; i++) {
        tmp_c = y[i];
        n[tmp_c]++;
    }
}

/// compute centers according to y
void CenterBasedClustering::Update_Cen() {
    // n = 0
    fill(n.begin(), n.end(), 0);
    // Cen = 0
    for (int k = 0; k < c_true; k++){
        fill(Cen[k].begin(), Cen[k].end(), 0);
    }

    int tmp_c = 0;
    for (int i = 0; i < num; i++) {
        tmp_c = y[i];
        n[tmp_c]++;
        transform(Cen[tmp_c].begin(), Cen[tmp_c].end(), X[i].begin(), Cen[tmp_c].begin(), plus<>());
    }

    #pragma omp parallel for
    for (int k = 0; k < c_true; k++){
        transform(Cen[k].begin(), Cen[k].end(), Cen[k].begin(), bind(divides<>(), placeholders::_1, n[k]));
    }
}

/// return ||X - Y*Cen||_F^2
double CenterBasedClustering::GetSumSquaredError(){
    Matrix_Fnorm_byRow(X, xnorm);
    Matrix_Fnorm_byRow(Cen, cen_norm);

    vector<double> dist = vector<double>(num, 0);

    #pragma omp parallel for
    for (int i = 0; i < num; ++i) {
        int tmp_c = y[i];
        dist[i] = xnorm[i] + cen_norm[tmp_c] - 2*inner_product(X[i].begin(), X[i].end(), Cen[tmp_c].begin(), (double) 0);
    }

    return accumulate(dist.begin(), dist.end(), (double) 0);
}

//void CenterBasedClustering::InitWithRandomCenter() {
//    std::vector<int> indices(num);
//    iota(indices.begin(), indices.end(), 0);
//    random_shuffle(indices.begin(), indices.end());
//    for (int i = 0; i < c_true; ++i) {
//        Cen[i] = X[indices[i]];
//    }
//}

/// initial y
void CenterBasedClustering::initial_y(){
    if (init_method == "random_y"){
        InitWithRandomAssignment();
    }
}
/// generate y randomly
void CenterBasedClustering::InitWithRandomAssignment() {
    generate(y.begin(), y.end(), [=](){return uniform_rand(el_);});
}
