#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <iterator>

#include "kdtree.h"
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

  // build k-d tree
  kdt::KDTree<Point> kdtree(coords);

  // build query
  const Point query(coords[0]);
	
  // k-nearest neigbors search example
  const int k = 10;
  const std::vector<int> knnIndices = kdtree.knnSearch(query, k);



  
  // Build NN table [cityId] -> [NN0, NN1, ......, NNK]
  int** nearest;
    
  nearest = (int**)malloc(coords.size()*sizeof(int*));
  for(unsigned int i = 0; i < coords.size(); ++i) {
    int* neigh = (int*)malloc(k*sizeof(int));

    // build query
    Point query(coords[i]);
    // k-nearest neigbors search
    const std::vector<int> knnIndices = kdtree.knnSearch(query, k+1); // k+1 because the first one is the point itself
    for (unsigned int j = 1; j < k+1; ++j) {
      neigh[j-1] = knnIndices[j];
    }
    nearest[i] = neigh;
    
  }

  for (int i = 0; i < coords.size(); i+=1000) {
    std::cout << "[" << i << "] => ["; 
    for (int j = 0; j < k-1; ++j) {
      std::cout << nearest[i][j] << ", ";
    }
    std::cout << nearest[i][k-1] << "]" << std::endl;
  }
	
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
    coords.push_back(Point(x*1000,y*1000));
  }
  
  return coords;
}

template<typename T>
bool is_in(std::vector<T> const &v, T val) {

    auto it = find (v.begin(), v.end(), val);
  
    if (it != v.end())
        return true;
 
    return false;
}
