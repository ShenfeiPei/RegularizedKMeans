#include "regularized_k_means.h"

#include <thread>
#include <utility>

RegularizedKMeans::RegularizedKMeans(){}

RegularizedKMeans::RegularizedKMeans(vector<vector<double>>& X, int c_true, string init_method, bool warm_start, int n_jobs, unsigned int seed):CenterBasedClustering(X, c_true, std::move(init_method), seed),
    warm_start_(warm_start),
    n_jobs_(n_jobs == -1 ? thread::hardware_concurrency() : n_jobs),
    costs_(static_cast<int>(X.size()), std::vector<double>(c_true)) {}

RegularizedKMeans::~RegularizedKMeans(){}

void RegularizedKMeans::opt(int rep, string type){
    Y = vector<vector<int>> (rep, vector<int>(num, 0));
    obj = vector<double>(rep, 0);

    if (type == "Hard"){
        for (int rep_i = 0; rep_i < rep; rep_i ++){
            obj[rep_i] = SolveHard();
            Y[rep_i] = y;
        }
    }else if (type == "Soft"){
        for (int rep_i = 0; rep_i < rep; rep_i ++){
            obj[rep_i] = Solve(0);
            Y[rep_i] = y;
        }
    }
}

double RegularizedKMeans::SolveHard() {
  return SolveHard(num / c_true, (num + c_true - 1) / c_true);
}

double RegularizedKMeans::SolveHard(int lower_bound, int upper_bound) {
  return Solve([this, lower_bound, upper_bound]() -> NetworkSimplex {
    NetworkSimplex ns = NetworkSimplex();
    ns.BuildHard(this->costs_, this->c_true, lower_bound, upper_bound);
    return ns;
  });
}

double RegularizedKMeans::Solve(int f_th) {
  return Solve([this, f_th]() -> NetworkSimplex {
    NetworkSimplex ns = NetworkSimplex();
    ns.Build(this->costs_, f_th);
    return ns;
  });
}

double RegularizedKMeans::Solve(function<NetworkSimplex()> builder) {
  initial_y();
//  cout << y[0] << ", " << y[1] << ", " << y[2] << endl;
  UpdateCostMatrix();
  std::vector<int> old_assignments;
  NetworkSimplex ns_solver = builder();
  ns_solver.Simplex();
  ns_solver.GetAssignments(y);
  do {
    old_assignments = y;
    Update_Cen();
    UpdateCostMatrix();
    if (warm_start_) {
      ns_solver.UpdateCosts(costs_);
    } else {
      ns_solver = builder();
    }
    ns_solver.Simplex();
    ns_solver.GetAssignments(y);
  } while (old_assignments != y);
  return GetSumSquaredError();
}

void RegularizedKMeans::UpdateCostMatrix() {
    if (n_jobs_ <= 1) {
        for (int i = 0; i < num; ++i) {
            for (int j = 0; j < c_true; ++j) {
                costs_[i][j] = CalDistance(X[i], Cen[j]);
            }
        }
    } else {
        std::vector<std::thread> threads(n_jobs_);
        for (int t = 0; t < n_jobs_; ++t) {
            threads[t] = std::thread(std::bind([this](int thread_idx) {
                for (int idx = thread_idx; idx < num * c_true; idx += n_jobs_) {
                    int i = idx / c_true;
                    int j = idx % c_true;
                    costs_[i][j] = CalDistance(X[i], Cen[j]);
                }
            }, t));
        }
        std::for_each(threads.begin(), threads.end(), [](std::thread& x) { x.join(); });
    }
}

double RegularizedKMeans::CalDistance(std::vector<double>& data1,
                           std::vector<double>& data2){
  double result = 0.0;
  for (int i = 0; i < data1.size(); ++i) {
    result += (data1[i] - data2[i]) * (data1[i] - data2[i]);
  }
  return result;
}