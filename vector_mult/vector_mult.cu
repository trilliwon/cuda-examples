#include<stdlib.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<string>

#define TILE_WIDTH 2   /* set TILE_WIDTH 16 for the evaluation! */


__global__ vector_mult(float* a, float* b, float* output, int input_size) {
     // a, b :  input matrix address
     // input_size : width, height of input matrix
     // all input, output matrices are vectorized
 
     int tx = threadIdx.x, ty = threadIdx.y;
     int bx = blockIdx.x,  by = blockIdx.y;
 
     int row = by * blockDim.y + ty;
     int col = bx * blockDim.x + tx;
     
     if(row>=input_size ||col>=input_size) { return; }

     float sum = 0.0f;

     // output = a x b
     for (int i = 0; i<input_size; i++) {
         sum += a[row * input_size + i] * b[i * input_size + col];
     }

     output[row * input_size + col] = sum;
}