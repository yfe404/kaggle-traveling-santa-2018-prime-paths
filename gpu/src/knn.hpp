#ifndef KNN_HPP
#define KNN_HPP

#include "problem.hpp"
#include "kdtree.hpp"

// k+1 because the first one is the point itself

template <typename T>
vector<int> get_knn(vector<City<T>> path, City<T> city, int k) {
    kdt::KDTree<City<T>> kdtree(path);
    auto knnIndices = kdtree.knnSearch(city, k+1);
    knnIndices.erase(knnIndices.begin());
    return knnIndices;
}

template <typename T>
vector<vector<int>> get_knn(vector<City<T>> path, int k) {
    kdt::KDTree<City<T>> kdtree(path);
    vector<vector<int>> neighbors_idxs;
    for (size_t i = 0; i < path.size(); ++i) {
        auto knnIndices = kdtree.knnSearch(path[i], k+1);
        knnIndices.erase(knnIndices.begin());
        neighbors_idxs.push_back(knnIndices);
    }
    return neighbors_idxs;
}

#endif