#include <iostream>

constexpr int block_size = 16; 

__global__ 
void vecload_vecadd_kernel(float* a, float* b, float* c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    float4 a4 = ((float4*)a)[i];
    float4 b4 = ((float4*)b)[i];
    float4 c4; 

    c4.x = a4.x + b4.x; 
    c4.y = a4.y + b4.y; 
    c4.z = a4.z + b4.z; 
    c4.w = a4.w + b4.w; 
    ((float4*)c)[i] = c4;
}

void vecload_vecadd(float* a_h, float* b_h, float* c_h, int n){
    int size = n * sizeof(float); 
    float *a_d, *b_d, *c_d; 

    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice); 
    
    // Kernel invocation
    vecload_vecadd_kernel<<<(n + (block_size-1)) / block_size, block_size>>>(a_d, b_d, c_d, n);

    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost); 

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

int main(){
    float a[] = {
        255, 255, 255   
    };

    float b[] = {
        1, 1, 1   
    };

    float c[3];
    vecload_vecadd(a, b, c, 3);

    for (float element : c) {
            std::cout << static_cast<float>(element) << " ";
    }
    return 0;
}