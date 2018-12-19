#include <iostream>
#include <chrono>

#include "problem.hpp"
#include "kdtree.hpp"
#include "k_opt.hpp"
#include "io.hpp"

// NOTE: We use floats since single-precision arithmetic is
// much faster than double precision on GPUs.
#define PRECISION float

// CUDA Kernels
// ------------

// Reuse distance_lp from problem.hpp
template <typename T>
__host__ __device__
T distance_l1(Coord<T> a, Coord<T> b);

template <typename T>
__host__ __device__
T distance_l2(Coord<T> a, Coord<T> b);

// Reuse two_opt_score from k_opt.hpp
template <typename T>
__host__ __device__
T two_opt_score(City<T>* path, int k, int l);

// Host Code
// ---------

template <typename T>
struct delta_t {
    int i;
    int j;
    T delta;
};

// CPU single-threaded 2-opt
template <typename T>
vector<City<T>> two_opt_pass_cpu(vector<City<T>> path, int k) {
    kdt::KDTree<City<T>> kdtree(path);

    // This is not a greedy 2-opt, instead (like the GPU impl.) we
    // search for the best 2-opt for each index, and then maximize
    // the set of non-overlapping 2-opt to apply.
    vector<delta_t<T>> results(path.size(), {0, 0, 0});

    for (int i = 1; i < path.size(); i++) {
        // k+1 because the first one is the point itself
        for (int j : kdtree.knnSearch(path[i], k+1)) {
            // TODO: Stop if neighbor well-placed
            T s = two_opt_score(&path[0], i, j);
            if (s < results[i].delta) {
                results[i] = {i, j, s};
            }
        }
    }

    // TODO: Maximize profit and apply to new_path

    return path;
}

// GPU multi-threaded 2-opt
template <typename T>
vector<City<T>> two_opt_pass_gpu(vector<City<T>> path, int k) {
    // Build k-d tree
    kdt::KDTree<City<T>> kdtree(path);

    // Build NN table
    // TODO

    // build query
    // const City<PRECISION> query(cities[0]);

    // // k-nearest neigbors search example
    // const int k = 25;
    // const std::vector<int> knnIndices = kdtree.knnSearch(query, k);

    // for (auto i : knnIndices) {
    //     cout << cities[i].xy.x << endl;
    // }

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

    // return 0;
    return path;
}

int main(int argc, char const *argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " CITIES PATH" << endl;
        return 1;
    }

    chrono::time_point<chrono::steady_clock> start;
    chrono::time_point<chrono::steady_clock> finish;
    cout.precision(17);

    cout << "Loading cities from " << argv[1] << "..." << endl;
    auto cities = read_cities<PRECISION>(argv[1]);
    cout << "Loaded " << cities.size() << " cities" << endl;

    cout << "Loading path from " << argv[2] << "..." << endl;
    auto path = read_path(cities, argv[2]);
    if (!is_valid(path.begin(), path.end())) {
        cout << "Input path is not valid !";
    }

    cout << "Input path score = " << score(path) << endl;

    cout << "2-opt pass (CPU)" << endl;
    start = chrono::steady_clock::now();
    auto new_path = two_opt_pass_cpu(path, 15);
    finish = chrono::steady_clock::now();
    cout << "Time: " << chrono::duration_cast<chrono::duration<double> >(finish - start).count() << " seconds" << endl;
    cout << "New score = " << score(new_path) << endl;

    // cout << "2-opt pass (GPU)" << endl;
    // auto new_path = two_opt_pass_gpu(path, 15);
    // cout << "New score = " << score(new_path) << endl;

    return 0;
}
