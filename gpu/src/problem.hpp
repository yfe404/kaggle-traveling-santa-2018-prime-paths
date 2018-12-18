#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include <cmath>
#include <vector>

using namespace std;

template <typename T>
struct Coord {
    T x;
    T y;
};

template <typename T>
struct City {
    int i;
    bool p;
    Coord<T> xy;
    // Compatibility with kdtree.hpp
    static const int DIM = 2;
    const T& operator[](size_t i) const {if (i==0) {return xy.x;} else {return xy.y;}}
};

template <typename T>
T distance_l1(Coord<T> a, Coord<T> b) {
    return abs(a.x-b.x) + abs(a.y-b.y);
}

template <typename T>
T distance_l2(Coord<T> a, Coord<T> b) {
    return sqrt(pow(a.x-b.x, 2) + pow(a.y-b.y, 2));
}

template <class InputIt>
bool is_valid(InputIt first, InputIt last) {
    return (first->i == 0) && (last->i == 0) && (last-first == 197770);
}

// TODO: Use iterators (but will be slower ?)
template <typename T>
T score(vector<City<T>> path, int start = 0) {
    T score = 0.0;
    for (size_t i = 0; i < path.size()-1; i++) {
        if (((i+start+1) % 10 == 0) && !path[i].p) {
            score += distance_l2(path[i].xy, path[i+1].xy)*1.1;
        } else {
            score += distance_l2(path[i].xy, path[i+1].xy);
        }
    }
    return score;
}

#endif