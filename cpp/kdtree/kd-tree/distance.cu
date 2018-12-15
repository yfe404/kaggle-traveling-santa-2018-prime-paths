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

    for (i=3;i<=sqrt((float)n);i+=2)
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


__global__
void distance(int N, double* all_distances, double** coords, int* path, int path_size) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double dist;
    for (unsigned int i = index; i < path_size-1; i+=stride) {
        dist = sqrt((float)(pow(coords[path[i]][0] - coords[path[i+1]][0], 2) + pow(coords[path[i]][1] - coords[path[i+1]][1], 2)));
        all_distances[i] = dist;
    }
    
}

std::vector<int> read_primes() {
  int n;
  std::vector<int> primes;

  for (int i = 0; i < 200000; ++i) {
    primes.push_back(is_prime(i));
  }

  return primes;
}


int main()
{
    int path_size = 200000;
    int *path;

    // loading prime numbers list
    auto primes = read_primes();



    cudaMallocManaged(&path, path_size*sizeof(int));
    for(unsigned int i = 0; i < path_size-1; ++i)
        path[i] = i;
    path[path_size-1] = 0;

    double** coords;

    cudaMallocManaged(&coords, (path_size-1)*sizeof(double));

    double total_distance = 0.0;
    double *all_distances;
    int N = path_size-1;

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    for (unsigned int i = 0; i < path_size - 1; ++i) {
        cudaMallocManaged(&coords[i], 2*sizeof(double));
	coords[i][0] = i;
	coords[i][1] = i;
	
    }

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&all_distances, N*sizeof(double));

    // Run kernel on the GPU
    distance<<<numBlocks, blockSize>>>(N, all_distances, coords, path, path_size);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    for (unsigned int i = 0; i < N; ++i) {
        if ( (i+1)%10 == 0 and !primes[path[i]] )
            total_distance += 1.1*all_distances[i];
	else
	    total_distance += all_distances[i];
    }

    cout << "Total Distance: " << total_distance << endl;
 
 return 0;
}