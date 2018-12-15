// Example program
#include <iostream>
#include <string>
#include <vector>
#include <cmath>


#include <array>

using namespace std;

int is_prime(int n){
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


class Point : public std::array<double, 2>
{
public:                                                                               
	static const int DIM = 2;
	Point() {}
	Point(double x, double y) { (*this)[0] = x; (*this)[1] = y; }
};


double distance(const vector<Point>& coords, int a, int b) {
    return sqrt( pow(coords[a][0] - coords[b][0], 2) + pow(coords[a][1] - coords[b][1], 2) );
}

double distance(const vector<Point>& coords, const vector<int>& path) {
    double dist = 0.0;
    
    for (unsigned int i = 0; i < path.size()-1; ++i) {
        if (((i+1)%10) == 0 && !is_prime(path[i]))
            dist += 1.1*distance(coords, path[i], path[i+1]);
        else
            dist += distance(coords, path[i], path[i+1]);
    }
    
    return dist;
}

double distance(const vector<Point>& coords, const vector<int>& path, unsigned int from, unsigned int to, bool reverse=false) { // starts at from and stops when at to
    double dist = 0.0;
    
    if (!reverse) {
        for (unsigned int i = from; i < to; ++i) {
            if (((i+1)%10) == 0 && !is_prime(path[i]))
                dist += distance(coords, path[i], path[i+1]);
            else
                dist += distance(coords, path[i], path[i+1]);
        }
    } else {
        for (unsigned int i = to; i != from; --i) {
            unsigned int kappa = (to - i) + from; // new position of path[i] 
            if (((kappa+1)%10) == 0 && !is_prime(path[i]))
                dist += 1.1*distance(coords, path[i], path[i-1]);
            else
                dist += distance(coords, path[i], path[i-1]);
        }
    }
    
    return dist;
}

struct delta_t {
  int i;
  int j;
  double delta; 
} ;

void two_opt_step(const vector<Point>& coords, delta_t* results, const vector<int>& path, int** nearest, bool*filled) {
    for(unsigned int i = 0; i < path.size()-2; ++i) { 
        for(unsigned int j = 0; j < 2; ++j) {
            int nn_j = nearest[path[i]][j];
            int pos_j = -1;
            int jj = 0;
            while(pos_j == -1) {
                if(path[jj] == nn_j) {
                    pos_j = jj;
                    break;
                }
                jj++;
            }
            if (pos_j <= i) continue;
            delta_t* delta;
            delta->i = i;
            delta->j = j;
            delta->delta = distance(coords, path, i, pos_j, false) - distance(coords, path, i, pos_j, true);
            if(!filled[i] && delta->delta > 0) {
                filled[i] = true;
                results[i] = *delta;
            } 
        }
    }
}

void two_opt(const vector<Point>& coords, const vector<int>& path, int** nearest) {
    //best = route
    bool improved = true;
    delta_t* results;
    bool* filled;
    results = (delta_t*)malloc((path.size() - 3)*sizeof(delta_t));
    filled = (bool*)malloc((path.size() - 3)*sizeof(bool));
    while(improved) {
        improved = false;
        
        two_opt_step(coords, results, path, nearest, filled); // after this step, results contains all the pairs that improve path 
        // choose a move in results
        // if a move is chosen, update path, set improved to true, compute/print new total_distance for debugging if necessary
        // else => return;
    }
}


int main()
{
    vector<int> path = {0, 1, 2, 3, 4, 5, 0};
    vector<Point> coords;
    int k = 2;

    for (unsigned int i = 0; i < path.size() - 1; ++i) {
        coords.push_back(Point(i,i));
    }
    
    int** nearest;
    
    nearest = (int**)malloc(coords.size()*sizeof(int*));
    for(unsigned int i = 0; i < coords.size(); ++i) {
        int* neigh = (int*)malloc(k*sizeof(int));
        nearest[i] = neigh;
    }
    
    nearest[0][0] = 1;
    nearest[0][1] = 2;
    nearest[1][0] = 0;
    nearest[1][1] = 2;
    nearest[2][0] = 1;
    nearest[2][1] = 3;
    nearest[3][0] = 2;
    nearest[3][1] = 4;
    nearest[4][0] = 3;
    nearest[4][1] = 5;
    nearest[5][0] = 3;
    nearest[5][1] = 4;
    
    
    cout << distance(coords, path[1], path[2]) << endl;
    cout << distance(coords, path[0], path[path.size()-1]) << endl;
    cout << distance(coords, path) << endl;
    cout << distance(coords, path, 0, path.size()-1) << endl;
    cout << distance(coords, path, 0, 3) + distance(coords, path, 3, path.size()-1) << endl;
    
    cout << distance(coords, path, 2, 4) << endl;
    cout << distance(coords, path, 2, 4, true) << endl;

    
    for (unsigned int i = 0; i < path.size(); ++i) {
        cout << path[i] << endl;   
    }
 
 return 0;
}
