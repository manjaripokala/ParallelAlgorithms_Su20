#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>

//Reading array A from input file inp.txt
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

//Asserts for GPU errors
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void global_count_range_bins_kernel(int * d_out, int * d_in, int size)
{
 
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
 
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

__global__ void shmem_count_range_bins_kernel(int * d_out, int * d_in, int size)
{
    extern __shared__ int sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;
 
    // load shared mem from global mem
    sdata[tid] = 0;
    __syncthreads();  
 
    // stride is the total number of threads in the grid
    // Using stride increases the performance and benefits with scalability & thread reusage
    int stride = blockDim.x * gridDim.x;
    
    // do counts in shared mem
    for (; myId < size; myId += stride)
    {
        atomicAdd(&(sdata[d_in[myId]/100]), 1);
        __syncthreads();               // make sure all adds at one stage are done!
    }
 
    // assumes that threads per block size is atleast 10
        atomicAdd(&d_out[tid], sdata[tid]);
}

//kernel to perform parallel prefix sum
//assumes only 1 block (1 block can be utilized since we have only 10 elements)
__global__ void prefixsum(int *d_out, int * d_in, int size)
{
  extern __shared__ int sh_mem[];
  
  int tid = threadIdx.x;
  int myId = blockIdx.x * blockDim.x + threadIdx.x;
 
  sh_mem[tid] = d_in[myId];
  
  __syncthreads();
 
  if (myId < size)
  {
      for (int d = 1; d < blockDim.x; d *=2)
      {
        if (tid >= d) {
          sh_mem[tid] += sh_mem[tid - d];
        }
        __syncthreads();
      }
  }
  d_out[myId] = sh_mem[tid];
}


//Function to call corresponding kernel based on memory usage
void count_bins(int * d_out, int * d_in,
    int size, bool usesSharedMemory)
    {
    const int maxThreadsPerBlock = 512;
    int threads = maxThreadsPerBlock;

    // handles non power of 2 inputs
    int blocks = ceil(float(size) / float(maxThreadsPerBlock));

    if (usesSharedMemory)
    {
        //fprintf(q2a, "shared kernel in count \n");
        shmem_count_range_bins_kernel<<<blocks, threads, threads * sizeof(int)>>>
            (d_out, d_in, size);
    }
    else
    {
        //fprintf(q2a, "global kernel in count \n");
        global_count_range_bins_kernel<<<blocks, threads>>>(d_out, d_in, size);
        gpuErrorCheck( cudaPeekAtLastError() );
        gpuErrorCheck( cudaDeviceSynchronize() );
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
        fprintf(q2a, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        fprintf(q2a, "Using device %d:\n", dev);
        fprintf(q2a, "%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
        devProps.name, (int)devProps.totalGlobalMem,
               (int)devProps.major, (int)devProps.minor,
               (int)devProps.clockRate);
    }


    // generate the input array on the host
    Array A = initArrayA();
    int * h_in = A.array;
    const int ARRAY_SIZE = A.size;
    const int ARRAY_BYTES = A.size * sizeof(int);
    
    fprintf(q2a, "Array size is %d\n", ARRAY_SIZE);


    // declare GPU memory pointers
    int * d_in, * d_out, * s_in, * s_out, *prefix_out, *prefix_in;;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &s_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, 10*sizeof(int));
    cudaMalloc((void **) &s_out, 10*sizeof(int));

    // allocate memory for prefix sum, it has only 10 buckets
    cudaMalloc((void **) &prefix_out, 10*sizeof(int));
    cudaMalloc((void **) &prefix_in, 10*sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    //Problem 2a - Using Global Memory to get counts
    fprintf(q2a,"Using Global Memory to get counts\n");    

    //fprintf(q2a, "Running global count\n");
    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    count_bins(d_out, d_in, ARRAY_SIZE, false);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2a, "Using Global memory - average time elapsed: %f\n", elapsedTime);

    // copy back the counts from GPU
    int b[10];
    cudaMemcpy(&b, d_out, 10*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++) {
      fprintf(q2a, "Global Memory counts returned by device B[%d]: %d\n", i, b[i]);
    }

    //Problem 2b - Using Shared Memory to get counts
    // transfer the input array to the GPU
    cudaMemcpy(s_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    fprintf(q2b, "Array size is %d\n", ARRAY_SIZE);
    fprintf(q2b,"Using Shared Memory to get counts\n");    

    //fprintf(q2b, "Running shared count\n");
    cudaEventRecord(start, 0);
    count_bins(s_out, s_in, ARRAY_SIZE, false);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2b, "Using Shared memory - average time elapsed: %f\n", elapsedTime);

    // copy back the counts from GPU
    int s[10];
    cudaMemcpy(&s, s_out, 10*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++) {
      fprintf(q2b, "Shared Memory counts returned by device B[%d]: %d\n", i, s[i]);
    }
    
    // Problem 2c - Using Parallel Prefix SUM to calculate C
    fprintf(q2c, "Array size is %d\n", ARRAY_SIZE);

    // transfer the input scan array to the GPU
    cudaMemcpy(prefix_in, b, 10 * sizeof(int), cudaMemcpyHostToDevice);
 
    fprintf(q2c, "Running Parallel Prefix Sum\n");
    cudaEventRecord(start, 0);
    prefixsum<<<1, 10, 10 * sizeof(int)>>>(prefix_out, prefix_in, 10);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2c, "Using Parallel Prefix Sum - average time elapsed: %f\n", elapsedTime);

    gpuErrorCheck( cudaPeekAtLastError() );
    gpuErrorCheck( cudaDeviceSynchronize() );
 
    // copy back the counts from GPU
    int c[10];
    cudaMemcpy(&c, prefix_out, 10*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++) {
      fprintf(q2c, "Parallel Prefix sum returned by device: %d\n", c[i]);
    }

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(prefix_out);
    cudaFree(prefix_in);
    cudaFree(s_in);
    cudaFree(s_out);

    return 0;
}


// Reference: https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell

