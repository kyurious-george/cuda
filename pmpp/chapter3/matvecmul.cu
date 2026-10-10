#include <iostream>

constexpr int block_size = 16; 

__global__ 
void matmul_kernel(float* b, float* c, float* a, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // if thread is in bounds
    if (i < n){
        float dp = 0; 
        for (int k = 0; k < n; ++k){
            dp += b[i*n + k] * c[k];
        }
        a[i] = dp;
    }
}

void matmul(float* b_h, float* c_h, float* a_h, int n){
    int size = n * sizeof(float); 
    float *a_d, *b_d, *c_d; 

    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size*size);
    cudaMalloc((void **) &c_d, size);

    cudaMemcpy(b_d, b_h, size*size, cudaMemcpyHostToDevice); 
    cudaMemcpy(c_d, c_h, size, cudaMemcpyHostToDevice); 

    dim3 dimGrid((size + (block_size-1)) / block_size, 1, 1);
    dim3 dimBlock(block_size, 1, 1);
    
    // Kernel invocation
    matmul_kernel<<<dimGrid, dimBlock>>>(b_d, c_d, a_d, n);

    cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost); 

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

}

int main() {
    constexpr int n = 2;

    float b[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float c[] = {
        7.0f, 8.0f
    };

    float output[n] = {};

    matmul(b, c, output, n);

    for (int i = 0; i < n; ++i) {
        std::cout << output[i] << " ";
        std::cout << "\n";
    }

    return 0;
}