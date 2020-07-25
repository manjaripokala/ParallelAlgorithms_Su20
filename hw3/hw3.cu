#include <stdio.h>
#include <stdlib.h>
#include Array.c





int main(void) {

    const int ARRAY_A_BYTES = a->used * sizeof(int);
    int * arrayA_out;

    // declare GPU memory pointers
    int * d_arrayA_in;
    int d_minValue;
    int * d_arrayA_out;
    // Number of GPU devices
    int nDevices;
    cudaGetDeviceCount(&nDevices)

    // allocate GPU memory
    cudaMalloc((void**) &d_arrayA_in, ARRAY_A_BYTES);
    cudaMalloc((void**) &d_arrayA_out, nDevices * sizeof(int));
    cudaMalloc((void**) &d_minValue, sizeof(int));

    // transfer the array to the GPU
    cudaMemcpy(d_arrayA_in, a->array , ARRAY_A_BYTES, cudaMemcpyHostToDevice);


    // launch the kernel
    minValue<<<nDevices, 1>>>(d_arrayA_out, d_arrayA_in);

    // copy back the result array to the CPU
    cudaMemcpy(arrayA_out, d_arrayA_out , nDevices * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i =0; i < nDevices; i++) {
        printf("min of %d/n", arrayA_out[i]);
    }
//    printf("%d",minValue);
    //free memeory for part a of problem 1
    cudaFree(d_arrayA_in);
    cudaFree(d_arrayA_out);
//    freeArray(&a);

    return 0;
}

__global__ void minValue(int * outputArray, int * inputArray){
    blockIdx.x
    int idx = threadIdx.x;

    float f = d_in[idx];
    d_out[idx] = f*f*f;
}
