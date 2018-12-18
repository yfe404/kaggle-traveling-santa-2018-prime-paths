#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include <cmath>

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
};

template <typename T>
double distance_l1(Coord<T> a, Coord<T> b) {
    return abs(a.x-b.x) + abs(a.y-b.y);
}

template <typename T>
double distance_l2(Coord<T> a, Coord<T> b) {
    return sqrt(pow(a.x-b.x, 2) + pow(a.y-b.y, 2));
}

#endif