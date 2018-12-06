#include <iostream>
#include <array>
#include <vector>
#include <cmath>
#include <fstream>

#include "kdtree.h"

// user-defined point type
// inherits std::array in order to use operator[]
class MyPoint : public std::array<double, 2>
{
public:

	// dimension of space (or "k" of k-d tree)
	// KDTree class accesses this member
	static const int DIM = 2;

	// the constructors
	MyPoint() {}
	MyPoint(double x, double y)
	{ 
		(*this)[0] = x;
		(*this)[1] = y;
	}
};


int main(int argc, char **argv)
{
	const int seed = argc > 1 ? std::stoi(argv[1]) : 0;
	srand(seed);

	// loading points

	std::vector<MyPoint> coords;

	int n;
	double x,y;
	std::ifstream infile("data/kaggle.coords");

	while(infile >> n) {
	  infile >> x >> y;
	  coords.push_back(MyPoint(x,y));
	}
	std::cout << "Successfully loaded " << coords.size() << " points." << std::endl;


	// build k-d tree
	kdt::KDTree<MyPoint> kdtree(coords);

	// build query
	const MyPoint query(0.5, 0.5);
	
	// k-nearest neigbors search
	const int k = 10;
	const std::vector<int> knnIndices = kdtree.knnSearch(query, k);

	for (auto i = knnIndices.begin(); i != knnIndices.end(); ++i)
	  std::cout << *i << ' ';
	
	return 0;
}
