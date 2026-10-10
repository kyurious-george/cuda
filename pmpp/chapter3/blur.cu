#include <iostream>

constexpr int blur_size = 1;
constexpr int block_size = 16; 

// The input image is encoded as unsigned chars [0, 255]
__global__ 
void blur_kernel(unsigned char* in, unsigned char* out, int width, int height){
    int col = blockDim.x * blockIdx.x + threadIdx.x; 
    int row = blockDim.y * blockIdx.y + threadIdx.y; 

    // if thread is in bounds
    if (col < width && row < height){
        // Get the 1D offset for the grayscale image
        int num_pixels = 0; 
        int pixel_value = 0; 

        for (int blur_row = -blur_size; blur_row < blur_size+1; ++blur_row){
            for (int blur_col = -blur_size; blur_col < blur_size+1; ++blur_col){
                int cur_row = row + blur_row; 
                int cur_col = col + blur_col;
                if (0 <= cur_row && cur_row < height && 0 <= cur_col && cur_col < width){
                    num_pixels += 1;
                    pixel_value += in[cur_row*width + cur_col];
                }
            }
        }
        
        out[row*width+col] = ((float)pixel_value / num_pixels);
    }
}

void blur(unsigned char* in_h, unsigned char * out_h, int width, int height){
    int n = width * height;
    int size = n * sizeof(unsigned char); 
    unsigned char *in_d, *out_d; 

    cudaMalloc((void **) &in_d, size);
    cudaMalloc((void **) &out_d, size);

    cudaMemcpy(in_d, in_h, size, cudaMemcpyHostToDevice); 

    dim3 dimGrid((width + (block_size-1)) / block_size, (height + (block_size-1)) / block_size);
    dim3 dimBlock(block_size, block_size, 1);
    
    // Kernel invocation
    blur_kernel<<<dimGrid, dimBlock>>>(in_d, out_d, width, height);

    cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost); 

    cudaFree(in_d);
    cudaFree(out_d);
}

int main() {
    unsigned char input[] = {
        0,   0,   0,   0,
        0, 255, 255,   0,
        0, 255, 255,   0,
        0,   0,   0,   0
    };

    unsigned char output[4 * 4];

    blur(input, output, 4, 4);

    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            std::cout << static_cast<int>(output[row * 4 + col]) << " ";
        }
        std::cout << "\n";
    }

    return 0;
}