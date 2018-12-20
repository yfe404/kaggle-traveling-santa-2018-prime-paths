#include <iostream>
#include <chrono>

#include "problem.hpp"
#include "k_opt.hpp"
#include "knn.hpp"
#include "io.hpp"

// NOTE: We use floats since single-precision arithmetic is
// much faster than double precision on GPUs.
#define PRECISION float

// CPU + GPU Functions
// -------------------

template <typename T>
__host__ __device__
T distance_l1(Coord<T> a, Coord<T> b);

template <typename T>
__host__ __device__
T distance_l2(Coord<T> a, Coord<T> b);

template <typename T>
__host__ __device__
T two_opt_score(City<T>* path, int k, int l);

template <typename T>
__host__ __device__
void two_opt_results(City<T>* path, int path_size, int** neighbors_idxs, int n_neighbors, delta_t<T>* results, int index, int stride);

// CUDA Kernels
// ------------

// /!\ Make sure results is initialized to an array of {0, 0, 0}
template <typename T>
__global__
void two_opt_pass_gpu_kernel(City<T>* path, int path_size, int** neighbors_idxs, int n_neighbors, delta_t<T>* results) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    two_opt_results(path, path_size, neighbors_idxs, n_neighbors, results, index, stride);
}

// Host Code
// ---------

// CPU single-threaded 2-opt
template <typename T>
vector<City<T>> two_opt_pass_cpu(vector<City<T>> path, int k) {
     // Build NN table [cityId] -> [NN0, NN1, ......, NNK]
     auto neighbors_idxs = get_knn(path, k);
     int** gpu_neighbors_idxs;
     cudaMallocManaged(&gpu_neighbors_idxs, path.size()*sizeof(int*));
 
     for (size_t i = 0; i < path.size(); ++i) {
         gpu_neighbors_idxs[i] = &neighbors_idxs[i][0];
     }

    //  auto neigh


    // Will contain the best move as a delta_t struct obj
    delta_t<T>* results = (delta_t<T>*)malloc((path.size()-3)*sizeof(delta_t<T>));
    for (size_t i = 0; i < path.size(); ++i) {
        results[i] = {0, 0, 0};
    }

    // vector<delta_t<T>> results(path.size(), {0, 0, 0});

    two_opt_results(&path[0], path.size(), neighbors_idxs, k, results, 0, 1);

    // TODO: Maximize profit and apply to new_path
    auto new_path(path);

    return new_path;
}

// GPU multi-threaded 2-opt
template <typename T>
vector<City<T>> two_opt_pass_gpu(vector<City<T>> path, int k) {
    // Build NN table [cityId] -> [NN0, NN1, ......, NNK]
    auto neighbors_idxs = get_knn(path, k);
    int** gpu_neighbors_idxs;
    cudaMallocManaged(&gpu_neighbors_idxs, path.size()*sizeof(int*));

    for (size_t i = 0; i < path.size(); ++i) {
        gpu_neighbors_idxs[i] = &neighbors_idxs[i][0];
    }

    // Two-Opt
    int blockSize = 64;
    int numBlocks = (path.size() + blockSize - 1) / blockSize;

    // Will contain the best move as a delta_t struct obj
    delta_t<T>* results;
    cudaMallocManaged(&results, (path.size()-3)*sizeof(delta_t<T>));
    for (size_t i = 0; i < path.size(); ++i) {
        results[i] = {0, 0, 0};
    }

    // Copy path to GPU
    // TODO: Use memcpy instead ?
    City<T>* gpu_path;
    cudaMallocManaged(&gpu_path, path.size()*sizeof(City<T>));
    for (size_t i = 0; i < path.size(); ++i) {
        gpu_path[i] = path[i];
    }

    // Call GPU kernel
    auto start = chrono::steady_clock::now();
    two_opt_pass_gpu_kernel<<<numBlocks, blockSize>>>(
        gpu_path, path.size(), gpu_neighbors_idxs, k, results
    );
    cudaDeviceSynchronize();

    auto finish = chrono::steady_clock::now();
    cout << "Time: " << chrono::duration_cast<chrono::duration<double> >(finish - start).count() << " seconds" << endl;

    
    // TODO: Maximize profit and apply to new_path
    auto new_path(path);

    return new_path;
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

    cout << "2-opt pass (GPU)" << endl;
    start = chrono::steady_clock::now();
    new_path = two_opt_pass_gpu(path, 15);
    finish = chrono::steady_clock::now();
    cout << "Time: " << chrono::duration_cast<chrono::duration<double> >(finish - start).count() << " seconds" << endl;
    cout << "New score = " << score(new_path) << endl;

    // NOTE: I removed the while(improved) loop from the two_opt_pass function,
    // I think it's better to handle this outside (like here).

    write_path(new_path.begin(), new_path.end(), "k_opt_" + to_string(score(new_path)) + ".csv");
    return 0;
}
