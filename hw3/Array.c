//
// Created by Pedzinski, Dale on 7/25/20.
//
typedef struct {
    int *array;
    size_t used;
    size_t size;
} Array;

//typedef struct {
//    char **array;
//    size_t used;
//    size_t size;
//} CharArray;

//typedef struct {
//    char **array;
//    size_t used;
//    size_t size;
//} CharArray;

void initArray(Array *a, size_t initialSize) {
    a->array = (int*) malloc(initialSize * sizeof(int));
    a->used = 0;
    a->size = initialSize;
}
//
//void initCharArray(CharArray *a, size_t initialSize) {
//    a->array = (char**) malloc(initialSize * sizeof(char[10]));
//    a->used = 0;
//    a->size = initialSize;
//}

void insertArray(Array *a, int element) {
    // a->used is the number of used entries, because a->array[a->used++] updates a->used only *after* the array has been accessed.
    // Therefore a->used can go up to a->size
    if (a->used == a->size) {
        a->size *= 2;
        a->array =(int*) realloc(a->array, a->size * sizeof(int));
    }
    a->array[a->used++] = element;
}

//void insertCharArray(CharArray *a, char* element) {
//    // a->used is the number of used entries, because a->array[a->used++] updates a->used only *after* the array has been accessed.
//    // Therefore a->used can go up to a->size
//    if (a->used == a->size) {
//        a->size *= 2;
//        a->array = (char**) realloc(a->array, a->size * sizeof(char[10]));
//    }
//    a->array[a->used++] = element;
//}
//
////void freeIntArray(IntArray *a) {
////    free(a->array);
////    a->array = NULL;
////    a->used = a->size = 0;
////}

Array initArrayA(){
    FILE *fp;
    char str[50000];
    Array a;
    initArray(&a, 100);

    /* opening file for reading */
    fp = fopen("inp.txt" , "r");
    if(fp == NULL) {
        printf("%s","error");
        return a;
    }
    if( fgets (str, 50000, fp)!=NULL ) {
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
//
//CharArray initCharArrayA(){
//    FILE *fp;
//    char str[50000];
//    CharArray a;
//    initCharArray(&a, 100);
//
//    /* opening file for reading */
//    fp = fopen("inp.txt" , "r");
//    if(fp == NULL) {
//        printf("%s","error");
//        return a;
//    }
//    if( fgets (str, 50000, fp)!=NULL ) {
//        /* writing content to stdout */
////        printf("%s\n", str);
//        char* token;
//        char* rest = str;
//
//        while ((token = strtok_r(rest, " , ", &rest)))
//            insertCharArray(&a, token);
//    }
//    fclose(fp);
//    return a;
//}




