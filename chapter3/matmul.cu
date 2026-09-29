#include <iostream>

constexpr int block_size = 16; 

// The input image is encoded as unsigned chars [0, 255]
// Each pixel is 3 consecutive chars for the 3 channels RGB
__global__ 
void matmul_kernel(float* a, float* b, float* c, int width, int height){
    int col = blockDim.x * blockIdx.x + threadIdx.x; 
    int row = blockDim.y * blockIdx.y + threadIdx.y; 

    // if thread is in bounds
    if (col < width && row < height){
        float dp = 0; 
        for (int k = 0; k < width; ++k){
            dp += a[row*width + k] * b[k*width + col];
        }
        c[row*width + col] = dp;
    }
}

void matmul(float* a_h, float* b_h, float* c_h, int width, int height){
    int n = width * height;
    int size = n * sizeof(float); 
    float *a_d, *b_d, *c_d; 

    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice); 

    dim3 dimGrid((width + (block_size-1)) / block_size, (height + (block_size-1)) / block_size);
    dim3 dimBlock(block_size, block_size, 1);
    
    // Kernel invocation
    matmul_kernel<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, width, height);

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

    float output[width * height] = {};

    matmul(a, b, output, width, height);

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            std::cout << output[row * width + col] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}