#include <stdio.h>
#include <cuda_runtime.h>
#include "Array.c"

__global__ void getlastdigit(int *d_out, int *d_in, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        d_out[index] = (d_in[index] % 10);
//    int idx = threadIdx.x;

}

int main(int argc, char **argv) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0) {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int) devProps.totalGlobalMem,
               (int) devProps.major, (int) devProps.minor,
               (int) devProps.clockRate);
    }
    Array A = initArrayA();
    int * h_in = A.array;
    const int ARRAY_SIZE = A.size;
    const int ARRAY_BYTES = A.size * sizeof(int);
    int h_out[ARRAY_SIZE];

    printf("array size is %d\n", ARRAY_SIZE);

    printf("%s\n", "Input[500]:");

    for (int i = 0; i < 500; i++) {
        printf("%d, ", (h_in[i] % 10) );
    }
    printf("\n%s\n", "Values[500]:");

    int *d_in;
    int *d_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);

    // transfer the array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // launch the kernel
    int M = 256;
    getlastdigit<<<(ARRAY_SIZE + M-1) / M,M>>>(d_out, d_in,ARRAY_SIZE );

    // copy back the result array to the CPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // print out the resulting array
    for (int i = 0; i < 500; i++) {
        printf("%d, ", h_out[i]);
    }
    cudaFree(d_in);
    cudaFree(d_out);
    printf("\n");
    return 0;
}