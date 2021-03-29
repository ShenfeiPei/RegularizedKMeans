cmake_minimum_required(VERSION 3.17)
project(Balanced_kmeans)

set(CMAKE_CXX_STANDARD 11)

add_executable(Balanced_kmeans main.cpp CenterBasedClustering.cpp k_means.cpp lasso_k_means.cpp network_simplex.cpp regularized_k_means.cpp)
