#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>

typedef struct {
    int *array;
    size_t used;
    size_t size;
} Array;

void initArray(Array *a, size_t initialSize) {
    a->array = (int*) malloc(initialSize * sizeof(int));
    a->used = 0;
    a->size = initialSize;
}

void insertArray(Array *a, int element) {
    if (a->used == a->size) {
        a->size += 1;
        a->array =(int*) realloc(a->array, a->size * sizeof(int));
    }
    a->array[a->used++] = element;
}
Array initArrayA(){
    FILE *fp;
    char str[50000];
    Array a;
    initArray(&a, 1);

    /* opening file for reading */
    fp = fopen("inp.txt" , "r");
    if(fp == NULL) {
        printf("%s","error");
        return a;
    }
    while( fgets (str, 50000, fp)!=NULL ) {
        /* writing content to stdout */
//        printf("%s\n", str);
        char* token;
        char* rest = str;

        while ((token = strtok_r(rest, " , ", &rest)))
            insertArray(&a, atoi(token));
    }
    fclose(fp);
    return a;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void global_count_range_bins_kernel(int * d_out, int * d_in, int size)
{

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    //printf("myId: %d", myId);
    // stride is the total number of threads in the grid
    // Using stride increases the performance and benefits with scalability & thread reusage
    int stride = blockDim.x * gridDim.x;

    // do counts in global mem
    for (; myId < size; myId += stride)
    {
        atomicAdd(&(d_out[d_in[myId]/100]), 1);
        __syncthreads();        // make sure all adds at one stage are done!
    }

}

void count_bins(int * d_out, int * d_intermediate, int * d_in,
                int size, bool usesSharedMemory)
{
    const int maxThreadsPerBlock = 512;
    int threads = maxThreadsPerBlock;
    // handles non power of 2 arrays
    int blocks = ceil(float(size) / float(maxThreadsPerBlock));

    if (usesSharedMemory)
    {
        printf("shared kernel in count \n");
        //shmem_count_range_bins_kernel<<<blocks, threads, threads * sizeof(int)>>>
        //    (d_out, d_in, size);
    }
    else
    {
        printf("global kernel in count \n");
        global_count_range_bins_kernel<<<blocks, threads>>>(d_out, d_in, size);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

}


int main(int argc, char **argv)
{
    FILE *q2a;
    FILE *q2b;
    FILE *q2c;
    q2a = fopen("q2a.txt", "w");
    q2b = fopen("q2b.txt", "w");
    q2c = fopen("q2c.txt", "w");

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


    // generate the input array on the host
    Array A = initArrayA();
    int * h_in = A.array;
    const int ARRAY_SIZE = A.size;
    const int ARRAY_BYTES = A.size * sizeof(int);

    printf("array size is %d\n", ARRAY_SIZE);

    // declare GPU memory pointers
    int * d_in, * d_intermediate, * d_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_intermediate, ARRAY_BYTES); // overallocated
    cudaMalloc((void **) &d_out, 10*sizeof(int));

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);


    // problem 2a - select 0 as kernel
    // problem 2a - select 1 as kernel
    int whichKernel = 0;
    if (argc == 2) {
        whichKernel = atoi(argv[1]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // launch the kernel
    switch(whichKernel) {
        case 0:
            printf("Running global count\n");
            cudaEventRecord(start, 0);
            count_bins(d_out, d_intermediate, d_in, ARRAY_SIZE, false);
            cudaEventRecord(stop, 0);
            break;
        case 1:
            printf("Running count with shared mem\n");
            cudaEventRecord(start, 0);
            count_bins(d_out, d_intermediate, d_in, ARRAY_SIZE, true);
            cudaEventRecord(stop, 0);
            break;
        default:
            fprintf(stderr, "error: ran no kernel\n");
            exit(EXIT_FAILURE);
    }
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);


    // copy back the bin counts from GPU
    int b[10];
    cudaMemcpy(&b, d_out, 10*sizeof(int), cudaMemcpyDeviceToHost);

    printf("average time elapsed: %f\n", elapsedTime);
    for(int i = 0; i < 10; i++) {
        printf("count returned by device B[%d]: %d\n", i, b[i]);
    }


    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);

    return 0;
}