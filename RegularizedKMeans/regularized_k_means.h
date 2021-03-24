#ifndef REGULARIZED_K_MEANS_H_
#define REGULARIZED_K_MEANS_H_

#include "../CenterBasedClustering/CenterBasedClustering.h"
#include "network_simplex.h"

class RegularizedKMeans : public CenterBasedClustering {
 public:
  vector<double> obj;
  RegularizedKMeans();
  RegularizedKMeans(vector<vector<double>>& data, int k, string init_method, bool warm_start, int n_jobs, unsigned int seed);
  ~RegularizedKMeans();
  double SolveHard();
  double SolveHard(int lower_bound, int upper_bound);
  double Solve(int f_th);
  void opt(int rep, string type);
 protected:
  double Solve(function<NetworkSimplex()> builder);
  void UpdateCostMatrix();
  bool warm_start_;
  int n_jobs_;
  vector<vector<double>> costs_;
  double CalDistance(std::vector<double>& data1, std::vector<double>& data2);
};

#endif  // REGULARIZED_K_MEANS_H_
