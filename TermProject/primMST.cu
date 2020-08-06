
#include <limits.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <iostream>
#include <queue>
#include <set>
#include<list>
#include<iterator>
#include <algorithm>


// Structure to represent a node and its distance
struct distNode { 
	int64_t node; 
	int64_t dist; 
  bool operator<(const distNode& rhs) const
    {
        return dist > rhs.dist || (dist == rhs.dist && node > rhs.node);;
    }
}; 

// Structure to represent a edge
struct edge { 
	int64_t from; 
	int64_t to; 
 	int64_t weight;
   	bool operator<(const edge& rhs) const
     {
         return weight > rhs.weight || (weight == rhs.weight && to > rhs.to);
     }
};

// Structure to represent a edge source & destination
struct fromTo { 
	int64_t from; 
	int64_t to; 
   	bool operator<(const fromTo& rhs) const
     {
         return to < rhs.to || (to == rhs.to && from < rhs.from);
     }
};


// Initialize global variables
std::vector<int64_t> parent; // Vector to store constructed MST 
std::vector<int64_t> dist; // dist values used to pick minimum weight edge in cut 
std::vector<bool> fixed; // Vector to store constructed MST 

std::priority_queue<distNode> H; //binary heap of (j,dist) initially empty;
std::set<int64_t> Q, R; //set of vertices initially empty;
std::set<fromTo> T; //{ set of edges } initially {};
std::set<fromTo> mwe; //set of edges; minimum weight edges for all vertices

// class to represent a graph object
class Graph
{
public:
	// construct a vector of vectors of distance Node pairs to represent an adjacency list
	std::vector<std::vector<edge>> adjList;

	// Graph Constructor
	Graph(std::vector<edge> const &edges, int64_t N)
	{
		// resize the vector to N elements of type vector<edge>
		adjList.resize(N);

		// add edges to the undirected graph
		for (auto &e: edges)
		{
			int64_t from = e.from;
			int64_t to = e.to;
			int64_t weight = e.weight;

			// insert at the end
			adjList[from].push_back(edge{from, to, weight});
			adjList[to].push_back(edge{to, from, weight});
		}
	}
};


// // A utility function to add an edge in an 
// // undirected graph. 
// void addEdge(std::vector<edge> adj[], int64_t u, int64_t v) 
// { 
//     adj[u].push_back(edge{u,v}); 
//     adj[v].push_back(edge{v,u}); 
// } 
  
// // A utility function to print the adjacency list 
// // representation of graph 
// void printGraph(std::vector<edge> adj[], int64_t V) 
// { 
//     for (int64_t v = 0; v < V; ++v) 
//     { 
//         printf("\n Adjacency list of vertex ");
//         printf("v :%d\n head ", v); 
//         for (auto x : adj[v]) 
//            printf("-> %d %d", x.to, x.from); 
//         printf("\n"); 
//     } 
// } 

// print adjacency list representation of graph
void printGraph(Graph const &graph, int64_t N)
{
	for (int64_t i = 0; i < graph.adjList.size(); i++)
	{
		// print all neighboring vertices of given vertex
		for (edge v : graph.adjList[i]){
			printf("( %d, %d, %d )", v.from, v.to, v.weight);
			fflush( stdout );
		}
		printf("\n");
		fflush( stdout );
	}
}

//Identifies all minimum weight edges for all vertices
void initMWE(Graph const &graph) 
{ 
	printf("In MWE\n");
	
	//while (!H.empty()) { 
	for (int64_t i = 0; i < graph.adjList.size(); i++) {
		// Extract the vertex with minimum dist value 
		int64_t prevWeight=INT_MAX;
		int64_t mint64_to, minFrom;
		// Iterate through all the vertices of graph 
		for (edge v : graph.adjList[i]) {
			// Get the Minimum weight edge for vertex v.from
			if (v.weight < prevWeight) { 
				mint64_to = v.to;
				minFrom = v.from;
				prevWeight = v.weight;
			}
		} 
		printf("insert from:%d, to:%d\n", minFrom, mint64_to);
		mwe.insert(fromTo{minFrom, mint64_to});
	}
} 

// Get Weight for an edge
int64_t getWeight(Graph const &graph, int64_t u, int64_t v) {
	int64_t weight;
	// Iterate through all adjacent vertices of u and extract weight of u to v edge
	for (edge adj : graph.adjList[u]) {
			//printf("In Getweight u:%d\n", u);
		// Get the Minimum weight edge for vertex v.from
		if (adj.to == v) { 
			printf("adj from:%d, to:%d, weight:%d\n", adj.from, adj.to, adj.weight);
			weight = adj.weight;
		}
	}
	return weight;
}

// Process Edge in Parallel
void processEdge1(Graph const &graph, int64_t z, int64_t k) 
{ 
	printf("processEdge - z:%d, k:%d\n", z, k);
	fflush( stdout );

	int64_t weight = getWeight(graph, z, k);
	printf("After getweight :%d\n", weight);
	fflush( stdout );
	//std::cout << mwe.find(fromTo{z, k})) <<std::flush;
	if (mwe.find(fromTo{z, k}) != mwe.end()) {
		printf("In MWE Find\n");
		fflush( stdout );
		fixed[k] = true;
		T.insert(fromTo{k, z}); // z is the parent of k
		if (R.insert(k).second) {
				printf("Insert success: %d\n", k);
		};
		printf("After R insert\n");
		fflush( stdout );
	}
	else if (dist[k] > weight) {
		printf("Not MWE and dist[%d]:%d, weight:%d\n", k, dist[k], weight);
		fflush( stdout );
		dist[k] = weight;
		parent[k] = z;
		if (Q.find(k) == Q.end()) {
			printf("%d Not in Q\n", k);
			fflush( stdout );
			Q.insert(k);
		}
	}


	std::set<int64_t>::iterator it; //set iterator 
	for (it=R.begin(); it!=R.end(); ++it) {
			printf("in R iterator ");
		int e = *it; 
		printf("%d\n", e); 
	}
}

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
// using Prim's algorithm 
//std::set<fromTo> primMST(Graph const &graph) 
void primMST(Graph const &graph) 
{ 
	printf("In PRIM\n");
  fflush( stdout );
	std::set<int64_t>::iterator it; //set iterator 
	

	// Initialize min heap with all vertices. dist value of 
	// all vertices (except 0th vertex) is initially infinite 

	// change 1000 to maximum value of vertex in out test graphs
	printf("parent size: %d\n", parent.size());
	printf("dist size: %d\n", dist.size());
	printf("fixed size: %d\n", fixed.size());

	// 	printf("Inside For:\n");
	// 		fflush( stdout );
 
	for(int i = 0; i < 1000; i ++) {
		parent.push_back(-1);
		dist.push_back(INT_MAX);
		fixed.push_back(false);
	}
	
	printf("parent size: %d\n", parent.size());
	printf("dist size: %d\n", dist.size());
	printf("fixed size: %d\n", fixed.size());
	fflush( stdout );
	
	//distNode d = H.top();
 	//H.pop();
	//printf("pop H: %d - %d\n", d.node, d.dist);

	// Make dist value of 0th vertex as 0 so that it 
	// is extracted first 
	dist[0] = 0; 
	H.push(distNode{0, dist[0]});
	// minHeap->array[0] = newdistNode(0, dist[0]); 
	// minHeap->pos[0] = 0; 

	initMWE(graph); //initialize minimum weight edges of given graph;

	// Initially size of min heap is equal to V 
	//minHeap->size = V; 
	printf("Heap size: %d\n", H.size());
	fflush( stdout );
	// In the following loop, min heap contains all nodes 
	// not yet added to MST. 
	//while (!H.empty()) { 
	for (int64_t i = 0; i < graph.adjList.size(); i++) {
		// Extract the vertex with minimum dist value 
		distNode d = H.top();
		H.pop();
		int64_t j = d.node; //pop the minimum distance vertex
		if (!fixed[j]) {
				printf("!fixed, j:%d\n", j);
				fflush( stdout );
			R.insert(j);
			fixed[j] = true;
			if (parent[j] != -1) {
					printf("Parent[%d] ! -1\n", j);
					fflush( stdout );
				T.insert(fromTo{j, parent[j]});
			}

			int64_t z;
			//while (!R.empty()) {
					//printf("Inside R loop, size\n");
					//fflush( stdout );
				//for (it=R.begin(); it!=R.end(); it++) {
				for (auto z: R) {
						printf("Inside R loop, size:%d\n", R.size());
						//fflush( stdout );
						if(R.find(2) != R.end()) {
								printf("true\n");
						}
					// call processEdge for all neighbors of vertex in R 
					//z = *it;
					printf("z in R: %d\n", z);
					fflush( stdout );
					R.erase(R.find(z));
					printf("After R erase \n");
					fflush( stdout );
					for (edge adj : graph.adjList[z]) {
						printf("adj from:%d, to:%d, weight:%d\n", adj.from, adj.to, adj.weight);
						fflush( stdout );
						int64_t k = adj.to; 
						if (!fixed[k]) {
								printf("!fixed, k:%d\n", k);
								fflush( stdout );
							processEdge1(graph, z, k);							 
						}
					}
				}	
			//}
			while (!Q.empty()) {
				for (it=Q.begin(); it!=Q.end(); ++it) {
					int64_t z = *it;
					Q.erase(it);
					printf("z in Q: %d\n", z);
					if (!fixed[z]) {
							printf("!fixedz[%d]\n", z);
						H.push(distNode{z, dist[z]});
					}
				}
			}
		}
	}
	//if (T.size() == graph.adjList.size() -1) {
	//	return T;
	//} else 
	//	return std::set<fromTo>{}; // return empty tree

} 

// Driver program to test above functions 
int main() 
{ 
	// vector of graph edges as per above diagram.
	// Please note that initialization vector in below format will
	// work fine in C++11, C++14, C++17 but will fail in C++98.
	std::vector<edge> edges;
	edges.push_back(edge{0,1,5});
	edges.push_back(edge{0,2,4});
	edges.push_back(edge{1,3,7});
	edges.push_back(edge{1,2,3});
	edges.push_back(edge{2,3,9});
	edges.push_back(edge{2,4,11});
	edges.push_back(edge{3,4,2});
	/*{
		// (x, y, w) -> edge from x to y having weight w
		{ 0, 1, 6 }, { 1, 2, 7 }, { 2, 0, 5 }, { 2, 1, 4 },
		{ 3, 2, 10 }, { 5, 4, 1 }, { 4, 5, 3 }
	};*/
	
	// Number of nodes in the graph
	int64_t N = 9;

	// construct graph
	Graph graph(edges, N);

	// print adjacency list representation of graph
	printGraph(graph, N);

	
	
  printf("Before Prim\n");
  fflush( stdout );
	
	primMST(graph); 

	
  printf("After Prim\n");
  fflush( stdout );


	// print edges of MST 
	//printMST(T); 

	std::set<fromTo>::iterator it; //set iterator
	printf("T size:%d\n", T.size());
	for (it=T.begin(); it!=T.end(); ++it) {
			printf("in iterator");
		fromTo e = *it; 
		printf("%d - %d\n", e.from, e.to); 
	}

	return 0; 
} 

//Reference: https://www.geeksforgeeks.org/prims-mst-for-adjacency-list-representation-greedy-algo-6/
// https://www.techiedelight.com/graph-implementation-using-stl/

