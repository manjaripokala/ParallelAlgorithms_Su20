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
__global__ void global_reduce_kernel(int * d_out, int * d_in, int size)
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
        __syncthreads();
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[myId];
    }
}

__global__ void shmem_reduce_kernel(int * d_out, int * d_in, int size)
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
    int blocks = ceil(float(size) / float(maxThreadsPerBlock));

    if (usesSharedMemory)
    {
        shmem_reduce_kernel<<<blocks, threads, threads * sizeof(int)>>>
                (d_intermediate, d_in, size);
    }
    else
    {
        global_reduce_kernel<<<blocks, threads>>>
                (d_intermediate, d_in,size);
    }

    // now we're down to one block left, so reduce it
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    if (usesSharedMemory)
    {
        shmem_reduce_kernel<<<blocks, threads, threads * sizeof(int)>>>
                (d_out, d_intermediate, size);
    }
    else
    {
        global_reduce_kernel<<<blocks, threads>>>
                (d_out, d_intermediate,size);
    }
}

__global__ void getlastdigit(int *d_out, int *d_in, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size)
        d_out[index] = (d_in[index] % 10);
//    int idx = threadIdx.x;

}

int main(int argc, char **argv)
{
    FILE *q1a;
    FILE *q1b;
    q1a = fopen("q1a.txt", "w");
    q1b = fopen("q1b.txt", "w");

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(q1a, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        fprintf(q1a,"Using device %d:\n", dev);
        fprintf(q1a,"%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem,
               (int)devProps.major, (int)devProps.minor,
               (int)devProps.clockRate);
    }
    Array A = initArrayA();
    int * h_in = A.array;
    const int ARRAY_SIZE = A.size;
    const int ARRAY_BYTES = A.size * sizeof(int);


//    int min = 0;
    fprintf(q1a,"array size is %d\n", ARRAY_SIZE);
//    clock_t t;
//    t = clock();
//    fprintf(q1a,"\n\nRUNNING PROBLEM 1 PART A\n\n" );
//    for(int i = 0; i < ARRAY_SIZE; i++) {
//        if (min > h_in[i]){
//            min = h_in[i];
//        }
//    }
//    t = clock() - t;
//    double time_taken = ((double)t)/(CLOCKS_PER_SEC/1000); // calculate the elapsed time
//    fprintf("The host took %f ms to execute\n", time_taken);
//    fprintf("min at host: %d\n", min);

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

    fprintf(q1a,"Running global reduce for min\n");
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, false);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    int h_out;
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    fprintf(q1a,"average time elapsed in ms: %f\n", elapsedTime);
    fprintf(q1a,"min returned by device: %d\n", h_out);

    fprintf(q1a,"Running reduce with shared mem\n");
    cudaEventRecord(start2, 0);
    reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, true);
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    float elapsedTime2;
    cudaEventElapsedTime(&elapsedTime2, start2, stop2);
    int h_out2;
    cudaMemcpy(&h_out2, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    fprintf(q1a,"average time elapsed in ms: %f\n", elapsedTime2);
    fprintf(q1a,"min returned by device: %d\n", h_out2);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);

//    fprintf("\n\nRUNNING PROBLEM 1 PART B\n\n" );

    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(q1b, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    dev = 0;
    cudaSetDevice(dev);

//    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0) {
        fprintf(q1b,"Using device %d:\n", dev);
        fprintf(q1b,"%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int) devProps.totalGlobalMem,
               (int) devProps.major, (int) devProps.minor,
               (int) devProps.clockRate);
    }

    int h2_out[ARRAY_SIZE];

    fprintf(q1b,"array size is %d\n", ARRAY_SIZE);

//    fprintf(q1b,"%s\n", "From Host:");
//
//    for (int i = 0; i < ARRAY_SIZE; i++) {
//            fprintf(q1b,"%d, ", (h_in[i] % 10));
//    }
    fprintf(q1b,"\n%s\n\n", "From Device:");

    int *d2_in;
    int *d2_out;

    // allocate GPU memory
    cudaMalloc((void **) &d2_in, ARRAY_BYTES);
    cudaMalloc((void **) &d2_out, ARRAY_BYTES);

    // transfer the array to the GPU
    cudaMemcpy(d2_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // launch the kernel
    int M = 256;
    getlastdigit<<<(ARRAY_SIZE + M-1) / M,M>>>(d2_out, d2_in,ARRAY_SIZE );

    // copy back the result array to the CPU
    cudaMemcpy(h2_out, d2_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // print out the resulting array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (i< ARRAY_SIZE) {
            fprintf(q1b,"%d, ", h2_out[i]);
        }
    }
    cudaFree(d2_in);
    cudaFree(d2_out);
    fprintf(q1b,"\n");

    return 0;
}