#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <iterator>
#include <fstream>
#include <sstream>
#include <cassert>



#include "kdtree.h"
#include "point.h"

using namespace std;

std::vector<int> read_primes();
std::vector<Point> load_coords();
int isPrime(int n);


double distance(double** coords, int a, int b) {
  return sqrt( (float)(pow(coords[a][0] - coords[b][0], 2) + pow(coords[a][1] - coords[b][1], 2) ));
}

double distance(double** coords, int* path, int path_size) {
    double dist = 0.0;
    
    for (unsigned int i = 0; i < path_size-1; ++i) {
        if (((i+1)%10) == 0 && !isPrime(path[i]))
            dist += 1.1*distance(coords, path[i], path[i+1]);
        else
            dist += distance(coords, path[i], path[i+1]);
    }
    
    return dist;
}


double distance(double** coords, int* path, int path_size, unsigned int from, unsigned int to, bool reverse=false) { // starts at from and stops when at to
    double dist = 0.0;
    
    if (!reverse) {
        for (unsigned int i = from; i < to; ++i) {
            if (((i+1)%10) == 0 && !isPrime(path[i]))
                dist += 1.1*distance(coords, path[i], path[i+1]);
            else
                dist += distance(coords, path[i], path[i+1]);
        }
    } else {
        for (unsigned int i = to; i != from; --i) {
            unsigned int kappa = (to - i) + from; // new position of path[i] 
            if (((kappa+1)%10) == 0 && !isPrime(path[i]))
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

void two_opt_step(double** coords, delta_t* result, int* path, int path_size, int** nearest, bool*filled) {
  
    for(unsigned int i = 0; i < path_size-2; ++i) {
      cout << i << endl;
        for(unsigned int j = 0; j < 10; ++j) {
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
	    delta_t delta;
            delta.i = i;
            delta.j = j;
            delta.delta = distance(coords, path, path_size, i, pos_j, false) - distance(coords, path, path_size, i, pos_j, true);
	    if(!(filled[i]) && delta.delta > 0) {
	      filled[i] = true;
	      result[i] = delta;
            } else if (filled[i] && delta.delta > result[i].delta) {
	      result[i] = delta;
            }
        }
    }
}
 

void two_opt(double** coords, int* path, int path_size, int** nearest) {
    //best = route
    bool improved = true;
    delta_t *result; // will contain the best move as a delta_t struct obj. 
    result = (delta_t*)malloc((path_size-3)*sizeof(delta_t));
    bool *filled; // tells wether a move that improves the score has been found or not
    filled = (bool*)malloc((path_size-3)*sizeof(bool));
    
    while(improved) {
        improved = false;
        
        two_opt_step(coords, result, path, path_size, nearest, filled); // after this step, results contains all the pairs that improve path 
        // choose a move in results
        // if a move is chosen, update path, set improved to true, compute/print new total_distance for debugging if necessary
        // else => return;
        for (int i = 0; i < (path_size-3); ++i){
            if (filled[i]) {
                std::cout << result[i].delta;
                break;
            }
        }
       
    }
}


std::vector<int> read_path(std::string filename) {

  std::string line;
  std::ifstream myfile (filename);
  vector<int> path;

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

  return path;
     
}



int main(int argc, char **argv)
{
  std::cout << std::fixed;
  std::cout << std::setprecision(2);

  const int seed = argc > 1 ? std::stoi(argv[1]) : 0;
  srand(seed);

  // loading prime numbers list
  auto primes = read_primes();
	
  // loading points
  auto coords_points = load_coords();

  double** coords;

  coords = (double**)malloc(coords_points.size()*sizeof(double*));
  for (unsigned int i = 0; i < coords_points.size(); ++i) {
    //        cudaMallocManaged(& coords[i], 2*sizeof(double));
    coords[i] = (double*)malloc(2*sizeof(double));
	coords[i][0] = coords_points[i][0];
	coords[i][1] = coords_points[i][1];
  }
  

  
  std::cout << "Successfully loaded " << coords_points.size() << " points." << std::endl;

  auto path = read_path("./1517078.tsp");
  int path_size = path.size();

  cout << distance(coords, path.data(), path_size) << endl;
  cout << distance(coords, path.data(), path_size, 0, path_size-1) << endl;
  
  // build k-d tree
  kdt::KDTree<Point> kdtree(coords_points);

  // build query
  const Point query(coords_points[0]);
	
  // k-nearest neigbors search example
  const int k = 10;
  const std::vector<int> knnIndices = kdtree.knnSearch(query, k);



  
  // Build NN table [cityId] -> [NN0, NN1, ......, NNK]
  int** nearest;
    
  nearest = (int**)malloc(coords_points.size()*sizeof(int*));
  for(unsigned int i = 0; i < coords_points.size(); ++i) {
    int* neigh = (int*)malloc(k*sizeof(int));

    // build query
    Point query(coords_points[i]);
    // k-nearest neigbors search
    const std::vector<int> knnIndices = kdtree.knnSearch(query, k+1); // k+1 because the first one is the point itself
    for (unsigned int j = 1; j < k+1; ++j) {
      neigh[j-1] = knnIndices[j];
    }
    nearest[i] = neigh;
    
  }

  for (int i = 0; i < coords_points.size(); i+=1000) {
    std::cout << "[" << i << "] => ["; 
    for (int j = 0; j < k-1; ++j) {
      std::cout << nearest[i][j] << ", ";
    }
    std::cout << nearest[i][k-1] << "]" << std::endl;
  }

  two_opt(coords, path.data(), path_size, nearest);
	
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
