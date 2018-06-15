/*
 * Find BLANK and replace your own code.
 * And submit report why do you replace the blank that way.
 */

 #include<stdlib.h>
 #include<iostream>
 #include<fstream>
 #include<vector>
 #include<string>
 
 #define TILE_WIDTH 2   /* set TILE_WIDTH 16 for the evaluation! */
 #define MAXPOOL_INPUT_FILENAME "input.txt"
 #define A_FILENAME "a.txt"
 #define B_FILENAME "b.txt"
 #define C_FILENAME "c.txt"
 
 using namespace std;
 

 __global__ void gemm(float *a, float *b, float *c, const float alpha, const float beta, float *output, const int input_size){
     // a, b, c : input matrix address
     // alpha, beta : input constant
     // output : output buffer address
     // input_size : width, height of input matrix
     // all input, output matrices are vectorized
 
     int tx = threadIdx.x, ty = threadIdx.y;
     int bx = blockIdx.x,  by = blockIdx.y;
 
     int row = by * blockDim.y + ty;
     int col = bx * blockDim.x + tx;
     
     if(row>=input_size ||col>=input_size) { return; }
     
     // allocate 2D tiles in __shared__ memory
     __shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
     __shared__ float s_b[TILE_WIDTH][TILE_WIDTH];
 
     float result = 0;
     
     // make sure you handle the case when the matrix sizes are not
     // multiple of TILE_WIDTH!
     // loop over the tiles of the input in phases
     for(int p = 0; p < ceilf(input_size/(float)TILE_WIDTH); ++p) {
         // CHANGE

         if (row < input_size && (p*TILE_WIDTH + tx) < input_size) {
             s_a[ty][tx] = a[p*input_size + p*TILE_WIDTH + tx];
         } else {
             s_a[ty][tx] = 0;
         }
     
        if (col < input_size && (p*TILE_WIDTH + ty) < input_size) {
            s_b[ty][tx] = b[(p*input_size + ty)*input_size + col];
        } else {
            s_b[ty][tx] = 0;
        }
         __syncthreads();
         // You need to use __syncthreads() a few times
         // to synchronize the threads in a thread block.

         for (int j = 0; j < TILE_WIDTH; j++) {
            result += s_a[ty][j] * s_b[j][tx];
         }
         
         __syncthreads();
		// after the entire tile's values have been used, proceed
     }

     // write out the result to output[row*input_size + col] 
     // CHANGE
     // boundary check
     if(row < input_size && col < input_size) {
        output[row * input_size + col] = result;
    }
}

int main(int argc, char **argv) {
    if(argc < 2) {
        cout << "usage : " << argv[0] << " alpha beta\n" << "example : " << argv[0] << " 100 0.5 0.8\n";
        return 1;
    }
    const int input_size = stoi(argv[1]);
    const float alpha = stof(argv[2]);
    const float beta = stof(argv[3]);

    // check input_size is power of 2
    if(input_size == 0 && (input_size & (input_size-1))){
        cout << "input_size must be power of 2\n";
        return 1;
    }

    float* a = new float[input_size*input_size];
    float* b = new float[input_size*input_size];
    float* c = new float[input_size*input_size];
    
    // read input matrices 
    ifstream a_in(A_FILENAME);
    ifstream b_in(B_FILENAME);
    ifstream c_in(C_FILENAME);

    for (int i = 0; i < input_size*input_size; ++i) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
        c[i] = rand() % 10;
    }
    
    // prints inputs for debugging.
    cout<<"\n\nalpha : "<<alpha<<'\n';
    cout<<"========== A ==========\n";
    for (int i = 0; i < input_size * input_size; ++i) {
        if(i%input_size==0) cout<<"\n";
        cout<<a[i]<<" ";
        if (input_size > 100) {
            cout << "\n.......";
            break;
        }
    }
    cout<<"\n========== B ==========\n";
    for (int i = 0; i < input_size * input_size; ++i) {
        if(i%input_size==0) cout<<"\n";
        cout<<b[i]<<" ";
        if (input_size > 100) {
            cout << "\n.......";
            break;
        }
    }
    cout<<"\n\nbeta : "<<beta<<'\n';
    cout<<"========== C ==========\n";
    for (int i = 0; i < input_size * input_size; ++i) {
        if(i%input_size==0) cout<<"\n";
        cout<<c[i]<<" ";
        if (input_size > 100) {
            cout << "\n.......";
            break;
        }
    }
    cout<<'\n';
    
    // set thread, block dimensions
    const dim3 block_size(TILE_WIDTH, TILE_WIDTH);
    const dim3 num_of_blocks(input_size/block_size.x+1, input_size/block_size.y+1);

    // memory allocation for the device
    float *dev_mem_a, *dev_mem_b, *dev_mem_c, *gemm_output;
    cudaMalloc(&dev_mem_a, sizeof(float) * input_size * input_size);
    cudaMalloc(&dev_mem_b, sizeof(float) * input_size * input_size);
    cudaMalloc(&dev_mem_c, sizeof(float) * input_size * input_size);
    cudaMalloc(&gemm_output, sizeof(float) * input_size * input_size);
    
    // copy variable to device memory
    cudaMemcpy(dev_mem_a, a, sizeof(float) * input_size * input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mem_b, b, sizeof(float) * input_size * input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mem_c, c, sizeof(float) * input_size * input_size, cudaMemcpyHostToDevice);

    // launch CUDA kernels

    // First launch gemm kernel
    gemm<<<num_of_blocks, block_size>>>(dev_mem_a, dev_mem_b, dev_mem_c, alpha, beta, gemm_output, input_size);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr, "ERROR %s\n", cudaGetErrorString(error));
        return 1;
    }

    // allocate output buf in main memory
    float *gemm_output_buf = (float*) malloc (sizeof(float)*input_size*input_size);
    
    // copy results from device to host
    cudaMemcpy(gemm_output_buf, gemm_output, sizeof(float)*input_size*input_size, cudaMemcpyDeviceToHost);
    
    // prints the results
    cout<<"\n========== GEMM OUTPUT ==========\n";
    for (int i = 0; i < input_size * input_size; ++i) {
        if(i%input_size==0) cout<<"\n";
        cout<<gemm_output_buf[i]<<" ";
    }

    cout<<'\n';

    cudaFree(dev_mem_a);
    cudaFree(dev_mem_b);
    cudaFree(dev_mem_c);
    cudaFree(gemm_output);
    free(gemm_output_buf);
    return 0;
}
