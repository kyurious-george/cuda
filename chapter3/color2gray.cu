#include <iostream>

constexpr int channels = 3; 
constexpr int block_size = 16; 

// The input image is encoded as unsigned chars [0, 255]
// Each pixel is 3 consecutive chars for the 3 channels RGB
__global__ 
void color2gray_kernel(unsigned char* in, unsigned char* out, int width, int height){
    int col = blockDim.x * blockIdx.x + threadIdx.x; 
    int row = blockDim.y * blockIdx.y + threadIdx.y; 

    if (col < width && row < height){
        // Get the 1D offset for the grayscale image
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * channels;
        
        unsigned char r = in[rgbOffset]; 
        unsigned char g = in[rgbOffset+1];
        unsigned char b = in[rgbOffset+2]; 
        
        out[grayOffset] = (unsigned char) 0.299f*r + 0.587f*g + 0.114f*b;
    }
}

void color2gray(unsigned char* in_h, unsigned char * out_h, int width, int height){
    int n = width * height;
    int in_size = n * channels * sizeof(unsigned char); 
    int out_size = n * sizeof(unsigned char); 
    unsigned char *in_d, *out_d; 

    cudaMalloc((void **) &in_d, in_size);
    cudaMalloc((void **) &out_d, out_size);

    cudaMemcpy(in_d, in_h, in_size, cudaMemcpyHostToDevice); 

    dim3 dimGrid((width + (block_size-1)) / block_size, (height + (block_size-1)) / block_size);
    dim3 dimBlock(block_size, block_size, 1);
    
    // Kernel invocation
    color2gray_kernel<<<dimGrid, dimBlock>>>(in_d, out_d, width, height);

    cudaMemcpy(out_h, out_d, out_size, cudaMemcpyDeviceToHost); 

    cudaFree(in_d);
    cudaFree(out_d);
}

int main(){
    unsigned char input[] = {
        255,   0,   0,   // red
        0, 255,   0,   // green
        0,   0, 255,   // blue
        255, 255, 255    // white
    };

   unsigned char output[4];
   color2gray(input, output, 2, 2);

   for (unsigned char element : output) {
        std::cout << static_cast<int>(element) << " ";
   }
   return 0;
}