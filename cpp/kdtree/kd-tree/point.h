#ifndef POINT_H
#define POINT_H

#include <array>

class Point : public std::array<double, 2>
{
public:

	// dimension of space (or "k" of k-d tree)
	// KDTree class accesses this member
	static const int DIM = 2;

	// the constructors
	Point() {}
	Point(double x, double y)
	{ 
		(*this)[0] = x;
		(*this)[1] = y;
	}
};

#endif
