#include <iostream>

constexpr int block_size = 16; 

__global__ 
void matmul_output_row_kernel(float* a, float* b, float* c, int width, int height){
    int col = blockDim.x * blockIdx.x + threadIdx.x; 

    if (col < width) {
        for (int row = 0; row < height; ++row) {
            float dp = 0; 
            for (int k = 0; k < width; ++k){
                dp += a[row*width + k] * b[k*width + col];
            }
            c[row*width + col] = dp;
        }
    }
}

__global__ 
void matmul_output_col_kernel(float* a, float* b, float* c, int width, int height){
    int row = blockDim.x * blockIdx.x + threadIdx.x; 

    if (row < height) {
        for (int col = 0; col < width; ++col) {
            // if thread is in bounds
                float dp = 0; 
                for (int k = 0; k < width; ++k){
                    dp += a[row*width + k] * b[k*width + col];
                }
                c[row*width + col] = dp;
        }
    }
}

void matmul(float* a_h, float* b_h, float* c_h, int width, int height, bool is_row_per_thread){
    int n = width * height;
    int size = n * sizeof(float); 
    float *a_d, *b_d, *c_d; 

    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice); 

    dim3 dimGrid((width + (block_size-1)) / block_size, 1, 1);
    dim3 dimBlock(block_size, 1, 1);
    
    // Kernel invocation
    if (is_row_per_thread) {
        matmul_output_row_kernel<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, width, height);
    } else {
        matmul_output_col_kernel<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, width, height);
    }

    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost); 

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

}

int main() {
    constexpr int width = 2;
    constexpr int height = 2;

    float a[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float b[] = {
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    float output1[width * height] = {};

    matmul(a, b, output1, width, height, true);

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            std::cout << output1[row * width + col] << " ";
        }
        std::cout << "\n";
    }

    float output2[width * height] = {};

    matmul(a, b, output2, width, height, false);

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            std::cout << output2[row * width + col] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}