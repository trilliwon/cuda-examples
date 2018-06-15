#include<stdlib.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include "gputimer.h"

#define TILE_WIDTH 2   /* set TILE_WIDTH 16 for the evaluation! */
#define MAXPOOL_INPUT_FILENAME "input.txt"

using namespace std;

__global__ void maxpool(float *input, float *output, const int input_size, const int filter_size) {
    // input : input_matrix address
    // output : output buffer address
    // input_size : width, height of input matrix
    // filter_size : filter_size of maxpooling
    // all input, output matrices are vectorized

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    int output_size = input_size / filter_size;

    // out of bound
    if (col >= output_size || row >= output_size) { return; }
    
    // 2D to 1D : (row, col) -> (row * N) + col
    float max_val = input[((row * filter_size) * input_size) + (col * filter_size)];

    for (int i = row * filter_size; i < row * filter_size + filter_size; i++) {
        for (int j = col * filter_size; j < col * filter_size + filter_size; j++) {
            // update max_val if needed
            max_val = fmaxf(max_val, input[(i * input_size) + j]);
        }
    }

    // assign max value
    output[(row * output_size) + col] = max_val;
}

int main(int argc, char **argv) {

    GpuTimer timer;
    if(argc < 2) {
        cout << "usage : " << argv[0] << " input_size filter_size alpha beta\n" << "example : " << argv[0] << " 100 2 0.5 0.8\n";
        return 1;
    }

    const int input_size = stoi(argv[1]);
    const int filter_size = stoi(argv[2]); // used for maxpooling
    const int maxpool_output_size = input_size/filter_size;

    // check input_size is power of 2
    if(input_size == 0 && (input_size & (input_size-1))){
        cout << "input_size must be power of 2\n";
        return 1;
    }

    if(filter_size == 0){
        cout << "filter_size cannot be 0\n";
        return 1;
    }

    float* maxpool_input = new float[input_size * input_size];
    
    // generate random array
    for (int i = 0; i < input_size*input_size; ++i) {
        maxpool_input[i] = rand() % 100;
    }
    
    // prints inputs for debugging.
    cout<<"filter size : "<<filter_size;
    cout<<"input size: " << input_size;
    cout<<"\n========== MAXPOOL_INPUT ==========\n";
    for (int i = 0; i < 100; ++i) {
        if(i%input_size==0) cout<<"\n";
        cout<<maxpool_input[i]<<" ";
    }
    if (input_size > 10 ) {
        cout << ".....";
    }
    cout<<'\n';

    // set thread, block dimensions
    const dim3 block_size(TILE_WIDTH, TILE_WIDTH);
    const dim3 num_of_maxpool_blocks(maxpool_output_size/block_size.x+1, maxpool_output_size/block_size.y+1);

    // memory allocation for the device
    float *dev_mem_input, *maxpool_output;
    cudaMalloc(&dev_mem_input, sizeof(float) * input_size * input_size);
    cudaMalloc(&maxpool_output, sizeof(float) * maxpool_output_size * maxpool_output_size);

    // copy variable to device memory
    cudaMemcpy(dev_mem_input, maxpool_input, sizeof(float) * input_size * input_size, cudaMemcpyHostToDevice);
    cudaError_t error = cudaGetLastError();
 
    timer.Start();

    // launch CUDA kernels
    // Then run maxpooling
    maxpool<<<num_of_maxpool_blocks, block_size>>>(dev_mem_input, maxpool_output, input_size, filter_size);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr, "ERROR %s\n", cudaGetErrorString(error));
        return 1;
    }
    timer.Stop();
    printf("Time elapsed = %g ms\n", timer.Elapsed());

    // allocate output buf in main memory
    float *maxpool_output_buf = (float*) malloc (sizeof(float)*maxpool_output_size*maxpool_output_size);
    
    // copy results from device to host
    cudaMemcpy(maxpool_output_buf, maxpool_output, sizeof(float)*maxpool_output_size*maxpool_output_size, cudaMemcpyDeviceToHost);
    
    // prints the results
    cout<<"\n========== MAXPOOL OUTPUT ==========\n";
    for (int i = 0; i < maxpool_output_size * maxpool_output_size; ++i) {
        if(i%maxpool_output_size==0) cout<<"\n";
        cout<<maxpool_output_buf[i]<<" ";
    }
    cout<<'\n';

    cudaFree(dev_mem_input);
    cudaFree(maxpool_output);
    free(maxpool_output_buf);
	delete[] maxpool_input;
    return 0;
}
