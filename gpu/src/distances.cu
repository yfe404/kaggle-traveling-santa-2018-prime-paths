// NOTE: I haven't validated those...

__global__
void distances_l1(float* out, City<float>* path, size_t path_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < path_size-1; i += stride) {
        out[i] = abs(path[i].xy.x-path[i+1].xy.x) + abs(path[i].xy.y-path[i+1].xy.y);
    }
}

__global__
void distances_l2(float* out, City<float>* path, size_t path_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < path_size-1; i += stride) {
        out[i] = sqrt(pow(path[i].xy.x-path[i+1].xy.x, 2) + pow(path[i].xy.y-path[i+1].xy.y, 2));
    }
}
