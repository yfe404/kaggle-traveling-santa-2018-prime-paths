#include <iostream>
#include <array>
#include <vector>
#include <cmath>

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

	// generate points
	const int npoints = 100;
	std::vector<MyPoint> points(npoints);
	for (int i = 0; i < npoints; i++)
	{
		const int x = rand();
		const int y = rand();
		points[i] = MyPoint(x, y);
	}

	// build k-d tree
	kdt::KDTree<MyPoint> kdtree(points);

	// build query
	const MyPoint query(0.5, 0.5);
	
	// k-nearest neigbors search
	const int k = 10;
	const std::vector<int> knnIndices = kdtree.knnSearch(query, k);

	for (auto i = knnIndices.begin(); i != knnIndices.end(); ++i)
	  std::cout << *i << ' ';
	
	return 0;
}
