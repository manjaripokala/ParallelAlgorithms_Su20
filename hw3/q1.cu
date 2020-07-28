#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

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
__global__ void global_reduce_kernel(int * d_out, int * d_in)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // do reduction in global mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if ( d_in[myId] > d_in[myId + s]){
                d_in[myId]= d_in[myId + s];
            }
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[myId];
    }
}

__global__ void shmem_reduce_kernel(int * d_out, const int * d_in)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if ( sdata[tid] > sdata[tid + s]){
                sdata[tid]= sdata[tid + s];
            }
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

void reduce(int * d_out, int * d_intermediate, int * d_in,
            int size, bool usesSharedMemory)
{
    // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock
    const int maxThreadsPerBlock = 512;
    int threads = maxThreadsPerBlock;
    int blocks = size / maxThreadsPerBlock;
    if (usesSharedMemory)
    {
        shmem_reduce_kernel<<<blocks, threads, threads * sizeof(int)>>>
                (d_intermediate, d_in);
    }
    else
    {
        global_reduce_kernel<<<blocks, threads>>>
                (d_intermediate, d_in);
    }

    // now we're down to one block left, so reduce it
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    if (usesSharedMemory)
    {
        shmem_reduce_kernel<<<blocks, threads, threads * sizeof(int)>>>
                (d_out, d_intermediate);
    }
    else
    {
        global_reduce_kernel<<<blocks, threads>>>
                (d_out, d_intermediate);
    }
}

__global__ void getlastdigit(int *d_out, int *d_in, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        d_out[index] = (d_in[index] % 10);
//    int idx = threadIdx.x;

}

int main(int argc, char **argv)
{
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

    // generate the input array on the host
    int min = 0;
    printf("array size is %d\n", ARRAY_SIZE);
    clock_t t;
    t = clock();
    printf("RUNNING PROBLEM 1 PART A\n", );
    for(int i = 0; i < ARRAY_SIZE; i++) {
        if (min > h_in[i]){
            min = h_in[i];
        }
    }
    t = clock() - t;
    double time_taken = ((double)t)/(CLOCKS_PER_SEC/1000); // calculate the elapsed time
    printf("The host took %f ms to execute\n", time_taken);
    printf("min at host: %d\n", min);

    // declare GPU memory pointers
    int * d_in, * d_intermediate, * d_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_intermediate, ARRAY_BYTES); // overallocated
    cudaMalloc((void **) &d_out, sizeof(int));

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    printf("Running global reduce\n");
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, false);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    int h_out;
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("average time elapsed in ms: %f\n", elapsedTime);
    printf("min returned by device: %d\n", h_out);

    printf("Running reduce with shared mem\n");
    cudaEventRecord(start2, 0);
    reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, true);
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    float elapsedTime2;
    cudaEventElapsedTime(&elapsedTime2, start2, stop2);
    int h_out2;
    cudaMemcpy(&h_out2, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("average time elapsed in ms: %f\n", elapsedTime2);
    printf("min returned by device: %d\n", h_out2);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);

    printf("RUNNING PROBLEM 1 PART B\n", );

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