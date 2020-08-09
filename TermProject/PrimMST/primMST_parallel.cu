%%cu
// C / C++ program for Prim's MST for adjacency list representation of graph 


#include <stdio.h> 
#include <queue>
#include <set>
#include <list>
#include <iterator>
#include <algorithm>


#define ARRAY_SIZE 15
//#define ARRAY_BYTES (15 * 15 * sizeof(int));
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


// Structure to represent a vertex and its distance
struct distNode { 
	int node; 
	int dist; 
  bool operator<(const distNode& rhs) const
    {
        return dist > rhs.dist || (dist == rhs.dist && node > rhs.node);;
    }
}; 

// Structure to represent an edge
struct edge { 
	int from; 
	int to; 
 	int weight;
   	bool operator<(const edge& rhs) const
     {
         return weight > rhs.weight || (weight == rhs.weight && to > rhs.to);
     }
};

// Structure to represent a edge source & destination
struct fromTo { 
	int from; 
	int to; 
   	bool operator<(const fromTo& rhs) const
     {
         return to < rhs.to || (to == rhs.to && from < rhs.from);
     }
};


// Initialize global variables
__device__ __managed__ int parent[ARRAY_SIZE]; // Vector to store parent nodes
__device__ __managed__ int dist[ARRAY_SIZE]; // dist values used to pick minimum weight edge in cut 
__device__ __managed__ bool fixed[ARRAY_SIZE]; // Vector to store flags for node traversal
std::vector<bool> nonEmptyIndices; // Vector to store non empty indices of vertices

std::priority_queue<distNode> H; //binary heap of (j,dist) initially empty;
__device__ __managed__ int Q[ARRAY_SIZE], R[ARRAY_SIZE]; //set of vertices initially empty;
//__device__ __managed__ std::set<int> R, Q;
__device__ __managed__ fromTo T[ARRAY_SIZE*ARRAY_SIZE]; //{ set of edges } initially {};
__device__ __managed__ fromTo mwe[ARRAY_SIZE*ARRAY_SIZE]; //set of edges; minimum weight edges for all vertices
__device__ __managed__ int z_device, Q_index, R_index; 

// class to represent a graph object
class Graph
{
public:
	// construct a vector of vectors of edges to represent an adjacency list
	std::vector<std::vector<edge>> adjList;
	
	//Graph Vectors
// std::vector<int> vertices; //Vector to hold nodes of graph
// std::vector<int> edges; //Vector to hold edges of graph
// std::vector<int> weights; //Vector to hold weights of graph

	// Graph Constructor
	Graph(std::vector<edge> const &edges, int N)
	{
		// resize the vector to hold upto vertex of maximum label value (elements of type vector<edge>)
		adjList.resize(N);
		// edges.resize(N);
		// weights.resize(N);
		nonEmptyIndices.resize(N);

		// add edges to the undirected graph
		for (auto &e: edges)
		{
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


// // A utility function to add an edge in an 
// // undirected graph. 
// void addEdge(std::vector<edge> adj[], int u, int v) 
// { 
//     adj[u].push_back(edge{u,v}); 
//     adj[v].push_back(edge{v,u}); 
// } 

// print adjacency list representation of graph
void printGraph(Graph const &graph)
{
	for (int i = 0; i < graph.adjList.size(); i++)
	{
		// print all neighboring vertices of given vertex
		for (edge v : graph.adjList[i]){
			printf("( %d, %d, %d )", v.from, v.to, v.weight);
		}
		printf("\n");
	}
}

//Delete element from array
template<typename T>
void deleteElement(T arr[], int n, int x) 
{ 
   // Search x in array 
   int i; 
   for (i=0; i<n; i++) 
      if (arr[i] == x) 
         break; 
  
   // If x found in array 
   if (i < n) 
   { 
     // reduce size of array and move all 
     // elements on space ahead 
     n = n - 1; 
     for (int j=i; j<n; j++) 
        arr[j] = arr[j+1]; 
   } 
  
   //return n; 
} 

template<typename T>
__device__ bool ifExist(T arr[], T val){
		for (int i=0; i<ARRAY_SIZE; i++) {
				if (arr[i] == val)
					return true;
		}
		return false;
}

__device__ bool ifExistMWE(fromTo arr[], fromTo ft){
		for (int i=0; i<ARRAY_SIZE*ARRAY_SIZE; i++) {
				if (arr[i].from == ft.from && arr[i].to == ft.to)
					return true;
		}
		return false;
}

template<typename T>
int getIndex(T arr[]){
		return sizeof(arr)/sizeof(arr[0]);
}

//Identifies all minimum weight edges for all vertices
void initMWE(Graph const &graph) 
{ 
	for (int i = 0; i < graph.adjList.size(); i++) {
		// Extract the vertex with minimum dist value 
		int prevWeight=INT_MAX;
		int min_to, minFrom;
		// Iterate through all the vertices of graph 
		//for (edge adj : graph.adjList[i]) {
		for (auto it=graph.adjList[i].begin(); it!=graph.adjList[i].end(); it++) {
			edge adj = *it;
			// Get the Minimum weight edge for vertex adj.from
			if (adj.weight < prevWeight) { 
				min_to = adj.to;
				minFrom = adj.from;
				prevWeight = adj.weight;
			}
		} 
		mwe[getIndex(mwe)] = fromTo{minFrom, min_to};
	}
} 

// Get Weight for an edge
int getWeight(Graph const &graph, int u, int v) {
	int weight;
	// Iterate through all adjacent vertices of u and extract weight of u to v edge
	for (edge adj : graph.adjList[u]) {
		// Get the Minimum weight edge for vertex v.from
		if (adj.to == v) { 			
			weight = adj.weight;
		}
	}
	return weight;
}


// Process Edge in Parallel
//__device__ void processEdge1(Graph const &graph, int z, int k)
// __device__ void processEdge1(int *allvertex_devicein, int *alledge_devicein, 
// 	int *allweight_devicein, int z_device, int k_device) 
// { 
// 	int weight;
// 	for(int i=0; i<ARRAY_SIZE*ARRAY_SIZE; i++) {
// 		if (allvertex_devicein[i] == z_device && alledge_devicein[i] == k_device) {
// 			weight = allweight_devicein[i];
// 		}
// 	}
// 	if (mwe.find(fromTo{z_device, k_device}) != mwe.end()) {
// 		fixed[k_device] = true;
// 		T.insert(fromTo{k_device, z_device}); // z is the parent of k
// 		R.insert(k_device);
// 	}
// 	else if (dist[k_device] > weight) {
// 		dist[k_device] = weight;
// 		parent[k_device] = z_device;
// 		if (Q.find(k_device) == Q.end()) {
// 			Q.insert(k_device);
// 		}
// 	}
// }
 
 
//Kernel to process edges in Parallel
__global__ void parallel_processEdge(int *allvertex_devicein, int *alledge_devicein, 
	int *allweight_devicein, int z_device, int R_index, int Q_index, int T_index)
{
 
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    printf("myId: %d", myId); 
    // stride is the total number of threads in the grid
    // Using stride increases the performance and benefits with scalability & thread reusage
    int stride = blockDim.x * gridDim.x;
    
    // do counts in global mem
    while (allvertex_devicein[myId] == z_device)
    {
			int k_device = alledge_devicein[myId];
			int w_device = allweight_devicein[myId];
        if (!fixed[k_device]) {
					if (ifExistMWE(mwe, fromTo{z_device, k_device})) {
						fixed[k_device] = true;
						T[T_index + myId] = fromTo{k_device, z_device}; // z is the parent of k
						R[R_index + myId - 1] = k_device;
					}
					else if (dist[k_device] > w_device) {
						dist[k_device] = w_device;
						parent[k_device] = z_device;

						if (!ifExist(Q, k_device)) {
							Q[Q_index+ myId - 1] = k_device;
						//if (Q.find(k_device) == Q.end()) {
						//	Q.insert(k_device);
						}
					}
					//processEdge1(allvertex_devicein, alledge_devicein, allweight_devicein, z_device, k_device);
		}
        __syncthreads();        // make sure all adds at one stage are done!
    }
}

//Kernel Setup
void kernel_setup(Graph const &graph, int z_device){
	
	const int ARRAY_BYTES = ARRAY_SIZE * ARRAY_SIZE * sizeof(int);
	printf("array bytes:%f\n", ARRAY_BYTES);

	//declare GPU pointers
	// generate the input array on the host
	//atmost a node can connect to all other nodes
	int allvertex_in[ARRAY_SIZE*ARRAY_SIZE], alledge_in[ARRAY_SIZE*ARRAY_SIZE], allweight_in[ARRAY_SIZE*ARRAY_SIZE];
	
	int j = 0;
	for (int i = 0; i < graph.adjList.size(); i++) {
		for(edge adj : graph.adjList[i]) {
			allvertex_in[j] = adj.from;
			alledge_in[j] = adj.to;
			allweight_in[j] = adj.weight;
			j++;
		}
	}

	for (int i = 0; i <ARRAY_SIZE*ARRAY_SIZE; i++) {
		printf("allvertex_in:%d, alledge_in:%d, allweight_in:%d\n", allvertex_in[i], alledge_in[i], allweight_in[i]);
	}

	// declare GPU memory pointers
    int * allvertex_devicein, * alledge_devicein, * allweight_devicein;

    // allocate GPU memory
    cudaMalloc((void **) &allvertex_devicein, ARRAY_BYTES);
	cudaMalloc((void **) &alledge_devicein, ARRAY_BYTES);
	cudaMalloc((void **) &allweight_devicein, ARRAY_BYTES);

	// transfer the input array to the GPU
	cudaMemcpy(allvertex_devicein, allvertex_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	gpuErrchk( cudaMemcpy(alledge_devicein, alledge_in, ARRAY_BYTES, cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(allweight_devicein, allweight_in, ARRAY_BYTES, cudaMemcpyHostToDevice) );


	cudaEvent_t start, stop;
    cudaEventCreate(&start);
	cudaEventCreate(&stop);

	printf("Running global reduce\n");
    cudaEventRecord(start, 0);
	
	parallel_processEdge<<<1, 8>>>
						(allvertex_devicein, alledge_devicein, allweight_devicein, z_device, getIndex(R), getIndex(Q), getIndex(T));
	gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

    cudaEventRecord(stop, 0);
	
	cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

		// free GPU memory allocation
	cudaFree(allvertex_devicein);
	cudaFree(alledge_devicein);
	cudaFree(allweight_devicein);
};
 

// A utility function used to print the constructed MST 
void printMST(std::set<fromTo> T) 
{ 
	std::set<fromTo>::iterator it; //set iterator
	for (it=T.begin(); it!=T.end(); it++) {
		fromTo e = *it; 
		printf("%d - %d\n", e.from, e.to); 
  }
} 

// The main function that constructs Minimum Spanning Tree (MST) 
// using Prim's Parallel algorithm 
fromTo* primMST(Graph const &graph, int N, int source) 
{ 
	std::set<int>::iterator it; //set iterator 
	

	// Initialize min heap with all vertices. dist value of 
	// all vertices (except 0th vertex) is initially infinite 
 
	for(int i = 0; i < N; i ++) {
		parent[i] = -1;
		dist[i] = INT_MAX;
		fixed[i] = false;
	}

	// Make distance value of source vertex as 0 so it is extracted first 
	dist[source] = 0; 
	H.push(distNode{source, dist[0]});

	initMWE(graph); //initialize minimum weight edges of given graph;

	// Loop for |V| - 1 iterations
	//while (!H.empty()) { 
	for (int i = 0; i < graph.adjList.size(); i++) {
		// Extract the vertex with minimum dist value 
		distNode d = H.top();
		H.pop();
		int j = d.node; //pop the minimum distance vertex
		if (!fixed[j]) {
			R[getIndex(R)] = j;
			fixed[j] = true;
			if (parent[j] != -1) {
				T[getIndex(T)] = fromTo{j, parent[j]};
			}

			
			while (getIndex(R) != 0){
					//if(R.find(2) != R.end()) {
						printf("true\n");
					//}
					// call processEdge for all neighbors of vertex in R 
					z_device = R[0];
					deleteElement(R, getIndex(R), z_device);
					//allocate pointers copy required inputs to device
					//int *z_device;
					//cudaMemcpy(fixed_device, fixed, ARRAY_BYTES, cudaMemcpyHostToDevice);
					//cudaMemcpy(z_device, z, sizeof(int), cudaMemcpyHostToDevice);
					//call kernel setup
					kernel_setup(graph, z_device);
					

					//copy back updated values from device
					//cudaMemcpy(&fixed, fixed_device, ARRAY_BYTES, cudaMemcpyDeviceToHost);

					// for (edge adj : graph.adjList[z]) {
					// 	int k = adj.to; 
					// 	if (!fixed[k]) {
					// 		processEdge1(graph, z, k);							 
					// }
				//}
			}	
			
			while (getIndex(Q) != 0) {
				for (int i = 0; i < getIndex(Q); i++) {
					int z = Q[i];
					deleteElement(Q, getIndex(Q), z);
					if (!fixed[z]) {
						H.push(distNode{z, dist[z]});
					}
				}
			}
		}
	}
	if (getIndex(T) == graph.adjList.size() -1) {
		return T;
	} else 
		return new fromTo[ARRAY_SIZE]; // return empty tree

} 

// Driver program to test above functions 
int main() 
{ 
	// vector of graph edges as per above diagram.
	// Please note that initialization vector in below format will
	// work fine in C++11, C++14, C++17 but will fail in C++98.
	std::vector<edge> edges;
	edges.push_back(edge{4,5,4});
	edges.push_back(edge{4,11,8});
	edges.push_back(edge{5,6,8});
	edges.push_back(edge{5,11,11});
	edges.push_back(edge{6,7,7});
	edges.push_back(edge{6,12,2});
	edges.push_back(edge{6,9,4});
	edges.push_back(edge{7,8,9});
	edges.push_back(edge{7,9,14});
	edges.push_back(edge{8,9,10});
	edges.push_back(edge{9,10,2});
	edges.push_back(edge{10,11,1});
	edges.push_back(edge{10,12,6});
	edges.push_back(edge{11,12,7});
		
	// Maxmum label value of vertices in the given graph, assume 1000
	//int N = 15;
	
	

	//create vertex, edge, weight arrays on host
	//const int ARRAY_SIZE = 15;
    //const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int) * ARRAY_SIZE;

	// construct graph
	Graph graph(edges, ARRAY_SIZE);

	// print adjacency list representation of graph
	printGraph(graph);

	//Source vertex as first non empty vertex in adjacency List
	int source;
	for(int i = 0; i<nonEmptyIndices.size(); i++) {
			if (nonEmptyIndices[i]) {
				source = i;
			break;
		}
	}

    // // generate the input array on the host
    // int allvertex_in[ARRAY_SIZE], alledge_in[ARRAY_SIZE], allweight_in[ARRAY_SIZE];
	// int i = 0;
	//uncomment this while reading from input file
    //for(int i = 0; i < ARRAY_SIZE; i++) {
	// for (auto &e: edges) {
    //     // generate input array of vertices, edges, weights
	// 	allvertex_in[i] = e.from;
	// 	alledge_in[i] = e.to;
	// 	allweight_in[i] = e.weight;

	// 	allvertex_in[i+] = e.from;
	// 	alledge_in[e.from] = e.to;
	// 	allweight_in[e.from] = e.weight;
    //     i++;
    // }
    // printf("count at host: %d\n", count);

	// declare GPU memory pointers
	//std::vector<std::vector<edge>> * adjList_devicein;
    //int * d_in, * d_intermediate, * d_out;

	
	
	printf("source:%d\n", source);
	
  printf("Before Prim\n");
  //fflush( stdout );
	
  primMST(graph, ARRAY_SIZE, source); 

  printf("After Prim\n");
  //fflush( stdout );

	printf("T size:%d\n", getIndex(T));
	printf("MST in iterator\n");
	for (int i =0; i<getIndex(T); i++) {
		fromTo e = T[i]; 
		printf("%d - %d\n", e.from, e.to); 
	}

	
	return 0; 
} 

//Reference: https://www.geeksforgeeks.org/prims-mst-for-adjacency-list-representation-greedy-algo-6/
// https://www.techiedelight.com/graph-implementation-using-stl/
