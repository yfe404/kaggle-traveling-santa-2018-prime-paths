#ifndef __SOLUTION_H__
#define __SOLUTION_H__


#include <vector>
#include <string>

#include "point.h"



class Solution {

  public:

  Solution(const std::vector<int>& cities): path(cities) {};
    Solution(const std::string filename);

    ~Solution() {  };

    double distance(const std::vector<Point>& coords, const std::vector<int>& primes, int start=0, int end=197769);

  private:
    std::vector<int> path;

};




#endif
