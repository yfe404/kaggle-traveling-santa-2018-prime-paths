#ifndef K_OPT_HPP
#define K_OPT_HPP

#include "problem.hpp"

template <typename T>
T two_opt_score(City<T>* path, int k, int l) {
    // before: a k ... l b
    // after:  a l ... k b
    int a = k-1;
    int b = l+1;

    T p_a   = !path[a].p && (a % 10 == 0) ? 1.1 : 1.0; // Before
    T p_l_b = !path[l].p && (l % 10 == 0) ? 1.1 : 1.0; // Before
    T p_k_b = !path[k].p && (l % 10 == 0) ? 1.1 : 1.0; // After

    T a_k = distance_l2(path[a].xy, path[k].xy) * p_a; // Before
    T a_l = distance_l2(path[a].xy, path[l].xy) * p_a; // After

    T l_b = distance_l2(path[l].xy, path[b].xy) * p_l_b; // Before
    T k_b = distance_l2(path[k].xy, path[b].xy) * p_k_b; // After

    T diff = (a_l - a_k) + (k_b - l_b);

    // penalties_diff = 0.0

    // for i in modrange(k, l-1, 10)
    //     penalties_diff +=
    //         (!path[l+k-i].p * distance(path[l+k-i].xy, path[l+k-i-1].xy)) -
    //         (!path[i].p * distance(path[i].xy, path[i+1].xy))
    // end

    // diff + penalties_diff*0.1
    return diff;
}

#endif