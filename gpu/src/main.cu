#include <iostream>

#include "problem.hpp"
#include "io.hpp"

__global__
void distances(float* out, City<float>* path, size_t path_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < path_size-1; i += stride) {
        out[i] = sqrt(pow(path[i].xy.x-path[i+1].xy.x, 2) + pow(path[i].xy.y-path[i+1].xy.y, 2));
    }
}

int main(int argc, char const *argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " CITIES PATH" << endl;
        return 1;
    }

    // NOTE: We use floats since single-precision arithmetic is
    // much faster than double precision on GPUs.

    cout << "Loading cities from " << argv[1] << "..." << endl;
    auto cities = read_cities<float>(argv[1]);
    cout << "Loaded " << cities.size() << " cities" << endl;

    cout << "Loading path from " << argv[2] << "..." << endl;
    auto path = read_path(cities, argv[2]);
    if (!is_valid(path.begin(), path.end())) {
        cout << "Input path is not valid !";
    }

    // Copy cities to unified memory
    cout << "Copying to unified memory..." << endl;
    City<float>* cuda_path;
    cudaMallocManaged(&cuda_path, path.size()*sizeof(City<float>));
    for (size_t i = 0; i < path.size(); i++) {
        cuda_path[i] = path[i];
    }

    int N = path.size() - 1;
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Compute distances on GPU
    float* distances_out;
    cudaMallocManaged(&distances_out, (path.size()-1)*sizeof(float));
    distances<<<blockSize, numBlocks>>>(distances_out, cuda_path, path.size());

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    for (size_t i = 0; i < path.size(); i++) {
        cout << distances_out[i] << endl;
    }

//   // build k-d tree
//   kdt::KDTree<Point> kdtree(coords_points);

//   // build query
//   const Point query(coords_points[0]);
	
//   // k-nearest neigbors search example
//   const int k = 25;
//   const std::vector<int> knnIndices = kdtree.knnSearch(query, k);

  
//   // Build NN table [cityId] -> [NN0, NN1, ......, NNK]
//   int** nearest;
    
//   cudaMallocManaged(&nearest, coords_points.size()*sizeof(int*));
//   for(unsigned int i = 0; i < coords_points.size(); ++i) {
//     int* neigh;
//     cudaMallocManaged(&neigh, k*sizeof(int));

//     // build query
//     Point query(coords_points[i]);
//     // k-nearest neigbors search
//     const std::vector<int> knnIndices = kdtree.knnSearch(query, k+1); // k+1 because the first one is the point itself
//     for (unsigned int j = 1; j < k+1; ++j) {
//       neigh[j-1] = knnIndices[j];
//     }
//     nearest[i] = neigh;
    
//   }



//   // Two-Opt
//   int blockSize = 64;
//   int numBlocks = (path_size + blockSize - 1) / blockSize;

  
//   bool improved = true;
//     delta_t *result; // will contain the best move as a delta_t struct obj. 
//     cudaMallocManaged(&result, (path_size-3)*sizeof(delta_t));
//     bool *filled; // tells wether a move that improves the score has been found or not
//     cudaMallocManaged(&filled, (path_size-3)*sizeof(bool));
    
//     while(improved) {
//         improved = false;
//         two_opt_step<<<numBlocks, blockSize>>>(coords, result, path_array, path_size, nearest, filled); // after this step, results contains all the pairs that improve path 
//         // choose a move in results
//         // if a move is chosen, update path, set improved to true, compute/print new total_distance for debugging if necessary
//         // else => return;
	
// 	 // Wait for GPU to finish before accessing on host
// 	cudaDeviceSynchronize();
//         for (int i = 0; i < (path_size-3); ++i){
//             if (filled[i]) {
//                 std::cout << result[i].delta;
//                 break;
//             }
//         }
       
//     }

    return 0;
}
