#include <iostream>
#include "CenterBasedClustering.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;

    vector<vector<double>> X(5, vector<double>(3, 0));
    unsigned int c = 2;
    string method = "random";
    unsigned int seed = 0;

    KMeans KM = KMeans(X, c, method, seed);
    KM.opt();

}
