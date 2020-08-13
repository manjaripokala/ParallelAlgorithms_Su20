
#include <stdio.h>
#include <queue>
#include <set>
#include <list>
#include <iterator>
#include <algorithm>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <limits.h>

#define NODE_ARRAY_SIZE 13
#define EDGE_ARRAY_SIZE 100
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// Structure to represent a vertex and its distance
struct distNode {
    int node;
    int dist;

    bool operator<(const distNode &rhs) const {
        return dist > rhs.dist || (dist == rhs.dist && node > rhs.node);;
    }
};

// Structure to represent an edge
struct edge {
    int from;
    int to;
    int weight;

    bool operator<(const edge &rhs) const {
        return weight > rhs.weight || (weight == rhs.weight && to > rhs.to);
    }
};

// Structure to represent an edge source & destination
struct fromTo {
    int from;
    int to;

    bool operator<(const fromTo &rhs) const {
        return to < rhs.to || (to == rhs.to && from < rhs.from);
    }
};


// Initialize global variables
__device__ __managed__ int parent[NODE_ARRAY_SIZE]; // Array to store parent nodes
__device__ __managed__ int dist[NODE_ARRAY_SIZE]; // Array to store node distances
__device__ __managed__ bool fixed[NODE_ARRAY_SIZE]; // Array to store flags for node traversal
std::vector<bool> nonEmptyIndices; // Array to store non empty indices of vertices

std::priority_queue <distNode> H; //binary heap of (j,dist) initially empty;
__device__ __managed__ int Q[NODE_ARRAY_SIZE], R[NODE_ARRAY_SIZE]; //set of vertices initially empty;
__device__ __managed__ fromTo T[EDGE_ARRAY_SIZE]; //{ set of edges } initially {};
__device__ __managed__ fromTo mwe[EDGE_ARRAY_SIZE]; //set of edges; minimum weight edges for all vertices
__device__ __managed__ int z_device, Q_index = 0, R_index = 0, mwe_index = 0, T_index = 0; //Indices to synchronize between host & device
__device__ __managed__ int edge_cnt = 0; //keeps track of #edges

//Arrays to hold all edges of a graph
int allvertex_in[EDGE_ARRAY_SIZE], alledge_in[EDGE_ARRAY_SIZE], allweight_in[EDGE_ARRAY_SIZE];


// class to represent a graph object
class Graph {
public:
    // construct a vector of vectors of edges to represent an adjacency list
    std::vector <std::vector<edge>> adjList;


    // Graph Constructor
    Graph(std::vector <edge> const &edges, int N) {
        // resize the vector to hold upto vertex of maximum label value (elements of type vector<edge>)
        //or assign labels to each vertex starting from 0
        adjList.resize(N);
        nonEmptyIndices.resize(N);

        // add edges to the undirected graph
        for (auto &e: edges) {
            int from = e.from;
            int to = e.to;
            int weight = e.weight;

            // insert at the end
            adjList[from].push_back(edge{from, to, weight});
            adjList[to].push_back(edge{to, from, weight});

            //flag the non empty indices in adjList
            nonEmptyIndices[from] = true;
            nonEmptyIndices[to] = true;
        }
    }
};


// print adjacency list representation of graph
void printGraph(Graph const &graph) {
    for (int i = 0; i < graph.adjList.size(); i++) {
        // print all neighboring vertices of given vertex
        for (edge v : graph.adjList[i]) {
            //printf("( %d, %d, %d )", v.from, v.to, v.weight);
        }
        //printf("\n");
    }
}

//Delete element from array
//template<typename T>
void deleteElement(int arr[], int arr_index, int size) {

    if (arr_index < size) {
        // decrease the size of array and move all elements ahead
        size = size - 1;
        for (int j = arr_index; j < size; j++)
            arr[j] = arr[j + 1];
    }
}

//Check if an element exists in an array
//template<typename T>
__device__ bool ifExist(int arr[], int val) {
    for (int i = 0; i < NODE_ARRAY_SIZE; i++) {
        if (arr[i] == val)
            return true;
    }
    return false;
}

__device__ bool ifExistMWE(fromTo arr[], fromTo ft) {
    for (int i = 0; i < edge_cnt; i++) {
        if (arr[i].from == ft.from && arr[i].to == ft.to)
            return true;
    }
    return false;
}


//Function to load edges into kernel pointer arrays
void load_kernelArrays(Graph const &graph) {
    // generate the input array on the host
    //atmost a node can connect to all other nodes
    for (int i = 0; i < graph.adjList.size(); i++) {
        for (edge adj : graph.adjList[i]) {
            allvertex_in[edge_cnt] = adj.from;
            alledge_in[edge_cnt] = adj.to;
            allweight_in[edge_cnt] = adj.weight;
            edge_cnt++;
        }
    }
}

//Identifies all minimum weight edges for all vertices
void initMWE(Graph const &graph) {
    for (int i = 0; i < graph.adjList.size(); i++) {
        int prevWeight = INT_MAX;
        int min_to, minFrom;
        // Iterate through all the vertices of graph
        for (auto it = graph.adjList[i].begin(); it != graph.adjList[i].end(); it++) {
            edge adj = *it;
            // Get the Minimum weight edge for vertex adj.from
            if (adj.weight < prevWeight) {
                min_to = adj.to;
                minFrom = adj.from;
                prevWeight = adj.weight;
            }
        }
        mwe[mwe_index] = fromTo{minFrom, min_to};
        mwe_index++;
    }
}


//Kernel to process edges in Parallel
__global__ void parallel_processEdge(int *allvertex_devicein, int *alledge_devicein,
                                     int *allweight_devicein, int z_device) {

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    // int tid  = threadIdx.x;

    //printf("block:%d, myId: %d\n", blockIdx.x, myId);

    // process edges in R
    if (myId < edge_cnt) {
        //printf("myId:%d, allvertex_devicein[myId] :%d\n", myId, allvertex_devicein[myId]);
        if (allvertex_devicein[myId] == z_device) {
            //printf("Z found, allvertex_devicein[myId] :%d\n", allvertex_devicein[myId]);
            ////printf("block:%d, myId: %d\n", blockIdx.x, myId);
            int k_device = alledge_devicein[myId];
            //printf("k_device: %d\n", k_device);
            int w_device = allweight_devicein[myId];
            //printf("w_device: %d\n", w_device);

            if (!fixed[k_device]) {
                if (ifExistMWE(mwe, fromTo{z_device, k_device})) {
                    //printf("In MWE and not fixed k, z:%d, k:%d\n", z_device, k_device);
                    fixed[k_device] = true;

                    int t = atomicAdd(&T_index, 1);
                    T[t] = fromTo{k_device, z_device}; // z is the parent of k

                    int r = atomicAdd(&R_index, 1);
                    R[r] = k_device;
                    //printf("R_index in kernel:%d\n", R_index);
                } else if (dist[k_device] > w_device) {
                    //printf("not minimum edge and not fixed k, z:%d, k:%d\n", z_device, k_device);
                    //printf("\n");
                    dist[k_device] = w_device;
                    parent[k_device] = z_device;

                    if (!ifExist(Q, k_device)) {
                        int q = atomicAdd(&Q_index, 1);
                        Q[q] = k_device;
                        //if (Q.find(k_device) == Q.end()) {
                        //	Q.insert(k_device);
                    }
                }
            }
            __syncthreads();        // make sure all updates are finished
        }
    }
}

//Kernel Setup
void kernel_setup(Graph const &graph, int z_device) {

    int threads = 512;
    int blocks = ceil(float(edge_cnt) / float(threads));

    const int ARRAY_BYTES = EDGE_ARRAY_SIZE * sizeof(int);
    //printf("array bytes:%f\n", ARRAY_BYTES);

    // declare GPU memory pointers
    int *allvertex_devicein, *alledge_devicein, *allweight_devicein;

    // allocate GPU memory
    cudaMalloc((void **) &allvertex_devicein, ARRAY_BYTES);
    cudaMalloc((void **) &alledge_devicein, ARRAY_BYTES);
    cudaMalloc((void **) &allweight_devicein, ARRAY_BYTES);

    // transfer the input array to the GPU
    cudaMemcpy(allvertex_devicein, allvertex_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
    gpuErrchk(cudaMemcpy(alledge_devicein, alledge_in, ARRAY_BYTES, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(allweight_devicein, allweight_in, ARRAY_BYTES, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    parallel_processEdge<<<blocks, threads>>>
            (allvertex_devicein, alledge_devicein, allweight_devicein, z_device);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
//    printf("Elpased Time:%f\n", elapsedTime);

    // free GPU memory allocation
    cudaFree(allvertex_devicein);
    cudaFree(alledge_devicein);
    cudaFree(allweight_devicein);
};


// Function to print the constructed MST
void printMST(std::set <fromTo> T) {
    std::set<fromTo>::iterator it; //set iterator
    for (it = T.begin(); it != T.end(); it++) {
        fromTo e = *it;
        printf("%d - %d\n", e.from, e.to);
    }
}

// The main function that constructs Minimum Spanning Tree (MST)
// using Prim's Parallel algorithm given in chapter 7
fromTo *primMST(Graph const &graph, int N, int source) {
    std::set<int>::iterator it; //set iterator

    // Initialize and assign dist value of
    // all vertices to 0 and source to infinite
    for (int i = 0; i < N; i++) {
        parent[i] = -1;
        dist[i] = INT_MAX;
        fixed[i] = false;
    }

    // Make distance value of source vertex as 0 so it is extracted first
    dist[source] = 0;
    H.push(distNode{source, dist[0]});

    initMWE(graph); //initialize minimum weight edges of given graph;

    // Loop for |V| - 1 iterations (our priority queue doesn't support decreaseKey so there will be duplicates in Heap H)
    for (int i = 0; i < graph.adjList.size(); i++) {
        // Extract the vertex with minimum dist value
        distNode d = H.top();
        H.pop();
        int j = d.node; //pop the minimum distance vertex
        //printf("Pop min distance node:%d\n", j);
        if (!fixed[j]) {
            R[R_index] = j;
            R_index++;
            fixed[j] = true;
            if (parent[j] != -1) {
                T[T_index] = fromTo{j, parent[j]};
                T_index++;
            }

            //printf("R_index: %d\n", R_index);
            while (R_index != 0) {
                // call processEdge for all neighbors of vertex in R
                //printf("R_index: %d\n", R_index);
                z_device = R[0];
                //printf("Z before kernel:%d\n", z_device);
                deleteElement(R, 0, NODE_ARRAY_SIZE);
                R_index--;
                //call kernel setup
                kernel_setup(graph, z_device);
            }

            while (Q_index != 0) {
                for (int i = 0; i < Q_index; i++) {
                    int z = Q[i];
                    //printf("z in Q:%d\n", z);
                    deleteElement(Q, i, NODE_ARRAY_SIZE);
                    Q_index--;
                    if (!fixed[z]) {
                        H.push(distNode{z, dist[z]});
                    }
                }
            }
        }
    }
    if (T_index == graph.adjList.size() - 1) {
        return T;
    } else
        return new fromTo[NODE_ARRAY_SIZE]; // return empty tree

}

const char *get_filename_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if (!dot || dot == filename) return "";
    return dot + 1;
}

// Driver program to call Prim
int main() {
    // vector of graph edges
    std::vector <edge> edges;
    edges.push_back(edge{0, 1, 866});
    edges.push_back(edge{0, 2, 187});
    edges.push_back(edge{0, 3, 399});

    edges.push_back(edge{1, 5, 605});
    edges.push_back(edge{1, 10, 1720});
    edges.push_back(edge{1, 11, 888});
    edges.push_back(edge{1, 12, 409});

    edges.push_back(edge{2, 1, 739});
    edges.push_back(edge{2, 3, 213});
    edges.push_back(edge{2, 4, 541});
    edges.push_back(edge{2, 5, 759});
    edges.push_back(edge{2, 6, 1416});
    edges.push_back(edge{2, 7, 1391});
    edges.push_back(edge{2, 8, 2474});
    edges.push_back(edge{2, 9, 2586});
    edges.push_back(edge{2, 10, 2421});
    edges.push_back(edge{2, 11, 1625});
    edges.push_back(edge{2, 12, 765});

    edges.push_back(edge{3, 4, 330});
    edges.push_back(edge{3, 5, 547});
    edges.push_back(edge{3, 12, 561});

    edges.push_back(edge{4, 5, 226});
    edges.push_back(edge{4, 6, 912});

    edges.push_back(edge{5, 6, 689});
    edges.push_back(edge{5, 7, 731});
    edges.push_back(edge{5, 11, 1199});
    edges.push_back(edge{5, 12, 213});

    edges.push_back(edge{6, 7, 224});
    edges.push_back(edge{6, 8, 1378});

    edges.push_back(edge{7, 8, 1234});
    edges.push_back(edge{7, 11, 641});
    edges.push_back(edge{7, 12, 631});

    edges.push_back(edge{8, 9, 337});
    edges.push_back(edge{8, 11, 861});

    edges.push_back(edge{9, 10, 678});
    edges.push_back(edge{9, 11, 967});

    edges.push_back(edge{10, 11, 1024});

    edges.push_back(edge{11, 12, 1013});

    struct dirent *de;  // Pointer for directory entry

    // opendir() returns a pointer of DIR type.
    DIR *dr = opendir(".");

    if (dr == NULL)  // opendir returns NULL if couldn't open directory
    {
        printf("Could not open current directory\n");
        return 0;
    }

    // Refer http://pubs.opengroup.org/onlinepubs/7990989775/xsh/readdir.html
    // for readdir()

    while ((de = readdir(dr)) != NULL) {
        char filePath[PATH_MAX + 1];
        if (strncmp(get_filename_ext(de->d_name), "csv", 2) == 0) {
//            printf("%s\n", de->d_name);
            realpath(de->d_name, filePath);
            printf("%s\n", filePath);
            char const *const fileName = filePath;
            FILE *file = fopen(fileName, "r"); /* should check the result */
//            printf("%s\n", "file opened");
            char line[4000];

            int state = 0;
            int subState = 0;
            int node_count = 0;

            while (fgets(line, sizeof(line), file)) {
                char *token;
                char *rest = line;
                int source;
                int target;
                int value;

//                printf("%s\n", "Next line:::");
                while ((token = strtok_r(rest, "|\n", &rest))) {
//                    printf("%s\n", token);
                    if (strncmp(token, "Nodes", 2) == 0) {
//                       printf("Section is %s\n", token);
                        state = 1;
                        subState = 0;
                        break;
                    }
                    if (strncmp(token, "Edges", 2) == 0) {
//                        printf("Section is %s\n", token);
                        state = 2;
                        subState = 0;
                        break;
                    }
                    if (strncmp(token, "Operator", 2) == 0) {
//                        printf("Section is %s\n", token);
                        state = 3;
                        subState = 0;
                        break;
                    }
                    if (state == 1 && subState == 0) {
//                        printf("Node is %s\n", token);
                        node_count = node_count + 1;
                        break;
                    }

                    if (state == 2 && subState == 0) {
                        //                            Source
//                        printf("Source is %s\n", token);
                        source = atoi(token);
                        subState = 1;
                    } else if (state == 2 && subState == 1) {
                        //                            Target
//                        printf("Target is %s\n", token);
                        target = atoi(token);
                        subState = 2;
                    } else if (state == 2 && subState == 2) {
                        //                            Value
//                        printf("Value is %s\n", token);
                        value = (int) atoi(token);
//                        edges.push_back(edge{source, target, value});
                        subState = 0;
                        break;
                    }
                    if (state == 3 && subState == 0) {
//                        printf("Airline is %s\n", token);
                        break;
                    }
                }
            }
            printf("close file\n");
            fclose(file);
//            printf("done with file\n");
            // Maxmum label value of vertices in the given graph, assume 1000
//            int N = node_count;
//            printf("%i count of nodes\n", node_count);
            // construct graph
            Graph graph(edges, NODE_ARRAY_SIZE);
            load_kernelArrays(graph);

            // print adjacency list representation of graph
//            printGraph(graph);

            //Source vertex as first non empty vertex in adjacency List
            //Or modify this to take from input file
            int source;
            for (int i = 0; i < nonEmptyIndices.size(); i++) {
                if (nonEmptyIndices[i]) {
                    source = i;
                    break;
                }
            }

            printf("source:%d\n", source);

            printf("Before Prim\n");

            primMST(graph, NODE_ARRAY_SIZE, source);

            printf("After Prim\n");
            fflush(stdout);
//            printf("MST in iterator\n");
            FILE *mst;
            char output[15], filename2[50],extension[5];

            strcpy(output,  "par_output_");
            strcpy(filename2, strtok(de->d_name, "."));
            strcpy(extension, ".txt");
            strcat(output, filename2);
            strcat(output, extension);
            mst = fopen(output, "w");
            //printf("T size:%d\n", T_index);
            //printf("MST in iterator\n");
            for (int i = 0; i < T_index; i++) {
                fromTo e = T[i];
                printf("%d - %d\n", e.from, e.to);
//                fprintf(mst,"%d - %d\n", e.from, e.to);
            }


            return 0;
        }
    }
//    return 0;
}

//Reference: https://www.geeksforgeeks.org/prims-mst-for-adjacency-list-representation-greedy-algo-6/
// https://www.techiedelight.com/graph-implementation-using-stl/