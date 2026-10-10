#include <iostream>

constexpr int tile_width = 16; 

__global__ 
void corner_turning_tiled_matmul_kernel(float* a, float* b, float* c, int width){
    // Add padding to the column so that we have an odd number and thereby using all the memory banks and avoiding as much conflict. 
    __shared__ float a_tile[tile_width][tile_width+1]; 
    __shared__ float b_tile[tile_width][tile_width+1];

    int y = blockIdx.y * tile_width + threadIdx.y; 
    int x = blockIdx.x * tile_width + threadIdx.x; 

    // loop over a and b in tiles
    float partial_sum = 0.0f;
    
    int tile_count = (width + tile_width - 1) / tile_width; 
    for (int t = 0; t < tile_count; ++t){
        int t_start = t*tile_width; 

        // coordinate loading of the threads into shared memory
        // a_tile uses the same format 
        if (y < width && (t_start + threadIdx.x) < width){
            a_tile[threadIdx.y][threadIdx.x] = a[y*width + t_start + threadIdx.x];
        } else {
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // b_tile requires corner turning so that when we read it's all coalesced
        if (t_start + threadIdx.y < width && x < width){
            b_tile[threadIdx.y][threadIdx.x] = b[(t_start + threadIdx.y)*width + x];
        } else {
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads(); 

        for (int k = 0; k < tile_width; ++k){
            partial_sum += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x]; 
        }
        __syncthreads(); 
    }
    if (y < width && x < width) {
        c[y*width + x] = partial_sum; 
    }
}

void matmul(float* a_h, float* b_h, float* c_h, int width){
    int n = width * width;
    int size = n * sizeof(float); 
    float *a_d, *b_d, *c_d; 

    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice); 

    dim3 dimGrid((width + (tile_width-1)) / tile_width, (width + (tile_width-1)) / tile_width);
    dim3 dimBlock(tile_width, tile_width, 1);
    
    // Kernel invocation
    corner_turning_tiled_matmul_kernel<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, width);

    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost); 

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

int main() {
    constexpr int width = 2;

    float a[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float b[] = {
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    float output[width * width] = {};

    matmul(a, b, output, width);

    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            std::cout << output[row * width + col] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}