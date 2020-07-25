#include <stdio.h>
#include <stdlib.h>


//__global__ void cube(float * d_out, float * d_in){
//    int idx = threadIdx.x;
//    float f = d_in[idx];
//    d_out[idx] = f*f*f;
//}

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
    // a->used is the number of used entries, because a->array[a->used++] updates a->used only *after* the array has been accessed.
    // Therefore a->used can go up to a->size
    if (a->used == a->size) {
        a->size *= 2;
        a->array =(int*) realloc(a->array, a->size * sizeof(int));
    }
    a->array[a->used++] = element;
}

void freeArray(Array *a) {
    free(a->array);
    a->array = NULL;
    a->used = a->size = 0;
}

int main(void) {

    FILE *fp;
    char str[50000];
    Array a;
    initArray(&a, 100);  // initially 5 elements

    /* opening file for reading */
    fp = fopen("inp.txt" , "r");
    if(fp == NULL) {
        perror("Error opening file");
        return(-1);
    }
    if( fgets (str, 50000, fp)!=NULL ) {
        /* writing content to stdout */
//        printf("%s\n", str);
        char* token;
        char* rest = str;

        while ((token = strtok_r(rest, " , ", &rest)))
            insertArray(&a, atoi(token));
    }
//    printf("%zu\n", a.used);
    fclose(fp);

//    freeArray(&a);

//    const int ARRAY_SIZE = 64;
//    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
//
//    // generate the input array on the host
//    float h_in[ARRAY_SIZE];
//    for (int i = 0; i < ARRAY_SIZE; i++) {
//        h_in[i] = float(i);
//    }
//    float h_out[ARRAY_SIZE];
//
//    // declare GPU memory pointers
//    float * d_in;
//    float * d_out;
//
//    // allocate GPU memory
//    cudaMalloc((void**) &d_in, ARRAY_BYTES);
//    cudaMalloc((void**) &d_out, ARRAY_BYTES);
//
//    // transfer the array to the GPU
//    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
//
//    // launch the kernel
//    cube<<<1, ARRAY_SIZE>>>(d_out, d_in);
//
//    // copy back the result array to the CPU
//    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
//
//    // print out the resulting array
//    for (int i =0; i < ARRAY_SIZE; i++) {
//        printf("%f", h_out[i]);
//        printf(((i % 4) != 3) ? "\t" : "\n");
//    }
//
//    cudaFree(d_in);
//    cudaFree(d_out);

    return 0;
}
