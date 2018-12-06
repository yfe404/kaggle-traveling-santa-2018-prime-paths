#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision

#include "kdtree.h"
#include "solution.h"
#include "point.h"

std::vector<int> read_primes();
std::vector<Point> load_coords();

int main(int argc, char **argv)
{
  std::cout << std::fixed;
  std::cout << std::setprecision(2);

  const int seed = argc > 1 ? std::stoi(argv[1]) : 0;
  srand(seed);

  // loading prime numbers list
  auto primes = read_primes();
	
  // loading points
  auto coords = load_coords();
  std::cout << "Successfully loaded " << coords.size() << " points." << std::endl;

  // load solution
  Solution solution("./data/path.tsp");
  std::cout << "Successfully loaded solution with score of " << solution.distance(coords, primes) << std::endl;

  // build k-d tree
  kdt::KDTree<Point> kdtree(coords);

  // build query
  const Point query(coords[0]);
	
  // k-nearest neigbors search
  const int k = 10;
  const std::vector<int> knnIndices = kdtree.knnSearch(query, k);

  for (auto i = knnIndices.begin(); i != knnIndices.end(); ++i)
    std::cout << *i << ' ';
	
  return 0;
}


int isPrime(int n){
    int i;

    if (n < 2)
      return 0;
    
    if (n==2)
        return 1;

    if (n%2==0)
        return 0;

    for (i=3;i<=sqrt(n);i+=2)
        if (n%i==0)
            return 0;

    return 1;
}

std::vector<int> read_primes() {
  int n;
  std::vector<int> primes;

  for (int i = 0; i < 200000; ++i) {
    primes.push_back(isPrime(i));
  }

  return primes;
}


std::vector<Point> load_coords() {

  int n;
  double x,y;
  std::vector<Point> coords;
  std::ifstream infile("data/kaggle.coords");

  while(infile >> n) {
    infile >> x >> y;
    coords.push_back(Point(x,y));
  }
  
  return coords;

}
