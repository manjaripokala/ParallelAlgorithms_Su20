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

__global__ void global_reduce_kernel(int * d_out, int * d_in, int size)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // do reduction in global mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && myId < size && myId+s < size)
        {
            d_in[myId] += d_in[myId + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[myId];
    }
}

__global__ void shmem_reduce_kernel(int * d_out, const int * d_in, int size)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ int sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && myId < size && myId+s < size)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

//kernel to create an intermediate array corresponding to given bins of array B
__global__ void global_count_range_bins_kernel(int * d_bins, int * d_binsin, int size, int x, int y)
{
 
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    // stride is the total number of threads in the grid
    // Using stride increases the performance and benefits with scalability & thread reusage
    int stride = blockDim.x * gridDim.x;
    
    // assign flags in global memory
    for (; myId < size; myId += stride)
    {
        if (d_binsin[myId] >= x && d_binsin[myId] <= y) {
            d_bins[myId] = 1;  
        } else {
            d_bins[myId] = 0; 
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

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
void reduce(int * d_out, int * d_intermediate, int * d_in,
    int size, bool usesSharedMemory)
{
    const int maxThreadsPerBlock = 512;
    int threads = maxThreadsPerBlock;
    
    // handles non power of 2 arrays
    int blocks = ceil(float(size) / float(maxThreadsPerBlock));

    if (usesSharedMemory)
    {
        shmem_reduce_kernel<<<blocks, threads, threads * sizeof(int)>>>
        (d_intermediate, d_in, size);
    }
    else
    {
        global_reduce_kernel<<<blocks, threads>>>
        (d_intermediate, d_in, size);
	gpuErrorCheck( cudaPeekAtLastError() );
    	gpuErrorCheck( cudaDeviceSynchronize() );
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
        (d_out, d_intermediate, size);
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
    
    fprintf(q2a, "array size is %d\n", ARRAY_SIZE);


    // declare GPU memory pointers
    int * d_in, * d_intermediate, * d_out, * d_bins, *d_binsin, *prefix_out, *prefix_in;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_binsin, ARRAY_BYTES);
    cudaMalloc((void **) &d_intermediate, ARRAY_BYTES); // overallocated
    cudaMalloc((void **) &d_bins, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, sizeof(int));
 
    // allocate memory for prefix sum, it has only 10 buckets
    cudaMalloc((void **) &prefix_out, 10*sizeof(int));
    cudaMalloc((void **) &prefix_in, 10*sizeof(int));

    const int maxThreadsPerBlock = 512;
    int threads = maxThreadsPerBlock;
    
    // handles non power of 2 arrays
    int blocks = ceil(float(ARRAY_SIZE) / float(maxThreadsPerBlock));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

//Problem 2a - Using Global Memory to get counts(from Reduction)
    fprintf(q2a,"Using Global Memory to get counts(from Reduction\n");
    // copy back the bin counts from GPU
    int b[10];
    // transfer the input array to the GPU
    gpuErrorCheck( cudaMemcpy(d_binsin, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice));

    //fprintf(q2a, "Running global count\n");

    //Bin 1
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 0, 99);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, false);
    gpuErrorCheck( cudaPeekAtLastError() );
    gpuErrorCheck( cudaDeviceSynchronize() );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2a, "Bin 1 - average time elapsed using global memory: %f\n", elapsedTime);
    cudaMemcpy(&b[0], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //Bin 2
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 100, 199);
    gpuErrorCheck( cudaPeekAtLastError() );
    gpuErrorCheck( cudaDeviceSynchronize() );

    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, false);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2a, "Bin 2 - average time elapsed using global memory: %f\n", elapsedTime);
    cudaMemcpy(&b[1], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //Bin 3
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 200, 299);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, false);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2a, "Bin 3 - average time elapsed using global memory: %f\n", elapsedTime);
    cudaMemcpy(&b[2], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //Bin 4
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 300, 399);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, false);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2a, "Bin 4 - average time elapsed using global memory: %f\n", elapsedTime);
    cudaMemcpy(&b[3], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //Bin 5
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 400, 499);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, false);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2a, "Bin 5 - average time elapsed using global memory: %f\n", elapsedTime);
    cudaMemcpy(&b[4], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //Bin 6
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 500, 599);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, false);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2a, "Bin 6 - average time elapsed using global memory: %f\n", elapsedTime);
    cudaMemcpy(&b[5], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //Bin 7
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 600, 699);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, false);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2a, "Bin 7 - average time elapsed using global memory: %f\n", elapsedTime);
    cudaMemcpy(&b[6], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //Bin 8
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 700, 799);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, false);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2a, "Bin 8 - average time elapsed using global memory: %f\n", elapsedTime);
    cudaMemcpy(&b[7], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //Bin 9
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 800, 899);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, false);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2a, "Bin 9 - average time elapsed using global memory: %f\n", elapsedTime);
    cudaMemcpy(&b[8], d_out, sizeof(int), cudaMemcpyDeviceToHost);
    
    //Bin 10
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 900, 999);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, false);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2a, "Bin 10 - average time elapsed using global memory: %f\n", elapsedTime);
    cudaMemcpy(&b[9], d_out, sizeof(int), cudaMemcpyDeviceToHost);


    for(int i = 0; i < 10; i++) {
      fprintf(q2a, "Global Memory - count returned by device: %d\n", b[i]);
    }


//Problem 2b - Using Shared Memory to get counts(from Reduction)
    fprintf(q2b,"Using Shared Memory to get counts(from Reduction\n");
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(q2b, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    dev = 0;
    cudaSetDevice(dev);

    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        fprintf(q2b, "Using device %d:\n", dev);
        fprintf(q2b, "%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
            devProps.name, (int)devProps.totalGlobalMem,
            (int)devProps.major, (int)devProps.minor,
            (int)devProps.clockRate);
    }

    fprintf(q2b,"array size is %d\n", ARRAY_SIZE);
    // copy back the bin counts from GPU
    int s[10];
    // transfer the input array to the GPU
    cudaMemcpy(d_binsin, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    //fprintf(q2b, "Running shared count\n");

    //Bin 1
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 0, 99);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, true);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2b, "Bin 1 - average time elapsed using shared memory: %f\n", elapsedTime);
    cudaMemcpy(&s[0], d_out, sizeof(int), cudaMemcpyDeviceToHost);
    
    //Bin 2
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 100, 199);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, true);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2b, "Bin 2 - average time elapsed using shared memory: %f\n", elapsedTime);
    cudaMemcpy(&s[1], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //Bin 3
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 200, 299);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, true);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2b, "Bin 3 - average time elapsed using shared memory: %f\n", elapsedTime);
    cudaMemcpy(&s[2], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //Bin 4
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 300, 399);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, true);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2b, "Bin 4 - average time elapsed using shared memory: %f\n", elapsedTime);
    cudaMemcpy(&s[3], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //Bin 5
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 400, 499);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, true);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2b, "Bin 5 - average time elapsed using shared memory: %f\n", elapsedTime);
    cudaMemcpy(&s[4], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //Bin 6
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 500, 599);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, true);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2b, "Bin 6 - average time elapsed using shared memory: %f\n", elapsedTime);
    cudaMemcpy(&s[5], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //Bin 7
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 600, 699);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, true);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2b, "Bin 7 - average time elapsed using shared memory: %f\n", elapsedTime);
    cudaMemcpy(&s[6], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //Bin 8
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 700, 799);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, true);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2b, "Bin 8 - average time elapsed using shared memory: %f\n", elapsedTime);
    cudaMemcpy(&s[7], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    //Bin 9
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 800, 899);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, true);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2b, "Bin 9 - average time elapsed using shared memory: %f\n", elapsedTime);
    cudaMemcpy(&s[8], d_out, sizeof(int), cudaMemcpyDeviceToHost);
    
    //Bin 10
    global_count_range_bins_kernel<<<blocks, threads>>>
        (d_bins, d_binsin, ARRAY_SIZE, 900, 999);
    cudaEventRecord(start, 0);
    reduce(d_out, d_intermediate, d_bins, ARRAY_SIZE, true);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(q2b, "Bin 10 - average time elapsed using shared memory: %f\n", elapsedTime);
    cudaMemcpy(&s[9], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    
    for(int i = 0; i < 10; i++) {
      fprintf(q2b, "Shared Memory - count returned by device: %d\n", s[i]);
    }


//Problem 2c - Using Parallel Prefix Scan 
    fprintf(q2c,"Using Parallel Prefix Scan to generate C\n");
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(q2c, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    dev = 0;
    cudaSetDevice(dev);

    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        fprintf(q2c, "Using device %d:\n", dev);
        fprintf(q2c, "%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
            devProps.name, (int)devProps.totalGlobalMem,
            (int)devProps.major, (int)devProps.minor,
            (int)devProps.clockRate);
    }

    // transfer the input scan array to the GPU
    cudaMemcpy(prefix_in, b, 10 * sizeof(int), cudaMemcpyHostToDevice);
 
    cudaEventRecord(start, 0);
    prefixsum<<<1, 10, 10 * sizeof(int)>>>(prefix_out, prefix_in, 10);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    gpuErrorCheck( cudaPeekAtLastError() );
    gpuErrorCheck( cudaDeviceSynchronize() );
 
    // copy back the prefix sum from GPU
    int c[10];
    cudaMemcpy(&c, prefix_out, 10*sizeof(int), cudaMemcpyDeviceToHost);

    fprintf(q2c, "Prefix Sum - average time elapsed: %f\n", elapsedTime);
    for(int i = 0; i < 10; i++) {
      fprintf(q2c, "Prefix Sum returned by device: %d\n", c[i]);
    }

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);
    cudaFree(d_bins);
    cudaFree(d_binsin);
    cudaFree(prefix_out);
    cudaFree(prefix_in);
 
    return 0;
}


// Reference: https://github.com/manjaripokala/sum20-Parallel-algs/blob/master/cuda-examples/reduce.cu
