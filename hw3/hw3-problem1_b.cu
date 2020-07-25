#include <stdio.h>
#include <cuda_runtime.h>
#include "Array.c"

// Find the first digit https://www.geeksforgeeks.org/find-first-last-digits-number/
//int firstDigit(int n)
//{
//    // Remove last digit from number
//    // till only one digit is left
//    while (n >= 10)
//        n /= 10;
//
//    // return the first digit
//    return n;
//}
//
//// Find the last digit
//int lastDigit(int n)
//{
//    // return the last digit
//    return (n % 10);
//}

__global__ void getlastdigit(int * d_out, int * d_in){
    int idx = threadIdx.x;
//    int x =;
    d_out[idx] = ( d_in[idx] % 10);
}

int main(int argc, char ** argv) {

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem,
               (int)devProps.major, (int)devProps.minor,
               (int)devProps.clockRate);
    }
    Array A = initArrayA();
    int * h_in = A.array;
    const int ARRAY_SIZE = A.size;
    const int ARRAY_BYTES = A.size * sizeof(int);
    int h_out[ARRAY_SIZE];

    printf("array size is %d\n", ARRAY_SIZE);
//    clock_t t;
//    t = clock();

    printf("%s\n", "Input[200]:");

    for(int i = 0; i < 200; i++) {
        printf("%d, ",(h_in[i]% 10));
    }
    printf("\n %s\n", "Values[200]:");

//    t = clock() - t;
//    double time_taken = ((double)t)/(CLOCKS_PER_SEC/1000); // calculate the elapsed time
//    printf("The host took %f ms to execute\n", time_taken);
//    printf("Max at host: %d\n", max);

    // declare GPU memory pointers
    int * d_in;
    int * d_out;

    // allocate GPU memory
    cudaMalloc((void**) &d_in, ARRAY_BYTES);
    cudaMalloc((void**) &d_out, ARRAY_BYTES);

    // transfer the array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // launch the kernel
    getlastdigit<<<1, ARRAY_SIZE>>>(d_out, d_in);

    // copy back the result array to the CPU
    cudaMemcpy(&h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // print out the resulting array
    for (int i =0; i < 200; i++) {
        printf("%d, ", h_out[i]);
    }
    printf("\n%s", "");
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}