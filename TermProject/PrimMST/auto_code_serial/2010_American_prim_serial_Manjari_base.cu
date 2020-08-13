%%cu
// C / C++ program for Prim's MST for adjacency list representation of graph 


#include <stdio.h> 
#include <queue>
#include <set>
#include <list>
#include <iterator>
#include <algorithm>

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
std::vector<int> parent; // Vector to store parent nodes
std::vector<int> dist; // dist values used to pick minimum weight edge in cut 
std::vector<bool> fixed; // Vector to store flags for node traversal
std::vector<bool> nonEmptyIndices; // Vector to store non empty indices of vertices

std::priority_queue<distNode> H; //binary heap of (j,dist) initially empty;
std::set<int> Q, R; //set of vertices initially empty;
std::set<fromTo> T; //{ set of edges } initially {};
std::set<fromTo> mwe; //set of edges; minimum weight edges for all vertices

// class to represent a graph object
class Graph
{
public:
	// construct a vector of vectors of edges to represent an adjacency list
	std::vector<std::vector<edge>> adjList;

	// Graph Constructor
	Graph(std::vector<edge> const &edges, int N)
	{
		// resize the vector to hold upto vertex of maximum label value (elements of type vector<edge>)
		adjList.resize(N);
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
			//printf("( %d, %d, %d )", v.from, v.to, v.weight);
		}
		//printf("\n");
	}
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
		mwe.insert(fromTo{minFrom, min_to});
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
void processEdge1(Graph const &graph, int z, int k) 
{ 
	int weight = getWeight(graph, z, k);

	if (mwe.find(fromTo{z, k}) != mwe.end()) {
		//printf("In MWE and not fixed k, z:%d, k:%d\n", z, k);

		fixed[k] = true;
		T.insert(fromTo{k, z}); // z is the parent of k
		R.insert(k);
	}
	else if (dist[k] > weight) {
		//printf("not minimum edge and not fixed k, z:%d, k:%d\n", z, k);
		dist[k] = weight;
		parent[k] = z;
		if (Q.find(k) == Q.end()) {
			Q.insert(k);
		}
	}
}

// A utility function used to print the constructed MST 
void printMST(std::set<fromTo> T) 
{ 
	std::set<fromTo>::iterator it; //set iterator
	for (it=T.begin(); it!=T.end(); it++) {
		fromTo e = *it; 
		////printf("%d - %d\n", e.from, e.to); 
  }
} 

// The main function that constructs Minimum Spanning Tree (MST) 
// using Prim's Parallel algorithm 
std::set<fromTo> primMST(Graph const &graph, int N, int source) 
{ 
	std::set<int>::iterator it; //set iterator 
	

	// Initialize min heap with all vertices. dist value of 
	// all vertices (except 0th vertex) is initially infinite 
 
	for(int i = 0; i < N; i ++) {
		parent.push_back(-1);
		dist.push_back(INT_MAX);
		fixed.push_back(false);
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
		//printf("Pop min distance node:%d\n", j);
		if (!fixed[j]) {
			R.insert(j);
			fixed[j] = true;
			if (parent[j] != -1) {
				T.insert(fromTo{j, parent[j]});
			}

			int z;
			while (!R.empty()) {
					//if(R.find(2) != R.end()) {
						////printf("true\n");
					//}
					// call processEdge for all neighbors of vertex in R 
					z = *R.begin();
					R.erase(R.find(z));
				for (edge adj : graph.adjList[z]) {
					int k = adj.to; 
					if (!fixed[k]) {
						processEdge1(graph, z, k);							 
					}
				}
			}	
			
			while (!Q.empty()) {
				for (it=Q.begin(); it!=Q.end(); ++it) {
					int z = *it;
					//printf("z in Q:%d\n", z);
					Q.erase(it);
					if (!fixed[z]) {
						H.push(distNode{z, dist[z]});
					}
				}
			}
		}
	}
	if (T.size() == graph.adjList.size() -1) {
		return T;
	} else 
		return std::set<fromTo>{}; // return empty tree

} 

// Driver program to test above functions 
int main() 

{ 
	// vector of graph edges as per above diagram.
	printf("2010_Alaskan_Serial\n");
	std::vector<edge> edges;
	
	// START

	// STOP
	
	// Maxmum label value of vertices in the given graph, assume 1000
	int N = 12000;
	
	// construct graph
	Graph graph(edges, N);

	// print adjacency list representation of graph
	//printGraph(graph);

	//Source vertex as first non empty vertex in adjacency List
	int source;
	for(int i = 0; i<nonEmptyIndices.size(); i++) {
			if (nonEmptyIndices[i]) {
				source = i;
			break;
		}
	}
	
	////printf("source:%d\n", source);
	
	cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        ////printf("Running global reduce\n");
        cudaEventRecord(start, 0);

  ////printf("Before Prim\n");
  //fflush( stdout );
	primMST(graph, N, source); 
  ////printf("After Prim\n");
  //fflush( stdout );

	 cudaEventRecord(stop, 0);

        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("Serial Elpased Time in ms:%f\n", elapsedTime);

	std::set<fromTo>::iterator it; //set iterator
	////printf("T size:%d\n", T.size());
	////printf("MST in iterator\n");
	for (it=T.begin(); it!=T.end(); ++it) {
		fromTo e = *it; 
		printf("%d - %d\n", e.from, e.to); 
	}

	return 0; 
} 

//Reference: https://www.geeksforgeeks.org/prims-mst-for-adjacency-list-representation-greedy-algo-6/
// https://www.techiedelight.com/graph-implementation-using-stl/
