#include <fstream>
#include <sstream>
#include <iterator>
#include <cassert>
#include <cmath>

#include "solution.h"


Solution::Solution(std::string filename) {

  std::string line;
  std::ifstream myfile (filename);

  if (myfile.is_open())
  {
    getline (myfile,line);
    myfile.close();
  }

  else throw std::string("Can't open file!");
   
  // Build an istream that holds the input string
  std::istringstream iss(line);

  // Iterate over the istream, using >> to grab floats
  // and push_back to store them in the vector
  std::copy(std::istream_iterator<float>(iss),
	    std::istream_iterator<float>(),
	    std::back_inserter(path));


  assert(path[0] == 0);
  assert(path[path.size()-1] == 0);
     
}


double Solution::distance(const std::vector<Point>& coords, const std::vector<int>& primes, int start, int end) {
  double distance = 0;
  int size = path.size();

  for (int i = start+1; i < size; ++i) {
    if ((i % 10 == 0) && !primes[path[i-1]]) {
      distance += sqrt(
        pow((coords[path[i-1]][0] - coords[path[i]][0]), 2) + 
	pow((coords[path[i-1]][1] - coords[path[i]][1]), 2)) * 1.1;
    } else {
      distance += sqrt(
        pow((coords[path[i-1]][0] - coords[path[i]][0]), 2) + 
	pow((coords[path[i-1]][1] - coords[path[i]][1]), 2));
    }
  }
    
  return distance;
}
