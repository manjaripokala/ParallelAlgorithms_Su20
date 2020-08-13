
#include <stdio.h> 
#include <queue>
#include <set>
#include <list>
#include <iterator>
#include <algorithm>

// Structure to represent a vertex and its distance
struct distNode { 
	int64_t node; 
	int64_t dist; 
  bool operator<(const distNode& rhs) const
    {
        return dist > rhs.dist || (dist == rhs.dist && node > rhs.node);;
    }
}; 

// Structure to represent an edge
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
std::vector<int64_t> parent; // Vector to store parent nodes
std::vector<int64_t> dist; // dist values used to pick minimum weight edge in cut 
std::vector<bool> fixed; // Vector to store flags for node traversal
std::vector<bool> nonEmptyIndices; // Vector to store non empty indices of vertices

std::priority_queue<distNode> H; //binary heap of (j,dist) initially empty;
std::set<int64_t> Q, R; //set of vertices initially empty;
std::set<fromTo> T; //{ set of edges } initially {};
std::set<fromTo> mwe; //set of edges; minimum weight edges for all vertices

// class to represent a graph object
class Graph
{
public:
	// construct a vector of vectors of edges to represent an adjacency list
	std::vector<std::vector<edge>> adjList;

	// Graph Constructor
	Graph(std::vector<edge> const &edges, int64_t N)
	{
		// resize the vector to hold upto vertex of maximum label value (elements of type vector<edge>)
		adjList.resize(N);
		nonEmptyIndices.resize(N);

		// add edges to the undirected graph
		for (auto &e: edges)
		{
			int64_t from = e.from;
			int64_t to = e.to;
			int64_t weight = e.weight;

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
// void addEdge(std::vector<edge> adj[], int64_t u, int64_t v) 
// { 
//     adj[u].push_back(edge{u,v}); 
//     adj[v].push_back(edge{v,u}); 
// } 

// print adjacency list representation of graph
void printGraph(Graph const &graph)
{
	for (int64_t i = 0; i < graph.adjList.size(); i++)
	{
		// print all neighboring vertices of given vertex
		for (edge v : graph.adjList[i]){
			printf("( %d, %d, %d )", v.from, v.to, v.weight);
		}
		printf("\n");
	}
}

//Identifies all minimum weight edges for all vertices
void initMWE(Graph const &graph) 
{ 
	for (int64_t i = 0; i < graph.adjList.size(); i++) {
		// Extract the vertex with minimum dist value 
		int64_t prevWeight=INT_MAX;
		int64_t min_to, minFrom;
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
int64_t getWeight(Graph const &graph, int64_t u, int64_t v) {
	int64_t weight;
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
void processEdge1(Graph const &graph, int64_t z, int64_t k) 
{ 
	int64_t weight = getWeight(graph, z, k);

	if (mwe.find(fromTo{z, k}) != mwe.end()) {
		printf("In MWE and not fixed k, z:%d, k:%d\n", z, k);

		fixed[k] = true;
		T.insert(fromTo{k, z}); // z is the parent of k
		R.insert(k);
	}
	else if (dist[k] > weight) {
		printf("not minimum edge and not fixed k, z:%d, k:%d\n", z, k);
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
		//printf("%d - %d\n", e.from, e.to); 
  }
} 

// The main function that constructs Minimum Spanning Tree (MST) 
// using Prim's Parallel algorithm 
std::set<fromTo> primMST(Graph const &graph, int64_t N, int64_t source) 
{ 
	std::set<int64_t>::iterator it; //set iterator 
	

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
	for (int64_t i = 0; i < graph.adjList.size(); i++) {
		// Extract the vertex with minimum dist value 
		distNode d = H.top();
		H.pop();
		int64_t j = d.node; //pop the minimum distance vertex
		printf("Pop min distance node:%d\n", j);
		if (!fixed[j]) {
			R.insert(j);
			fixed[j] = true;
			if (parent[j] != -1) {
				T.insert(fromTo{j, parent[j]});
			}

			int64_t z;
			while (!R.empty()) {
					//if(R.find(2) != R.end()) {
						//printf("true\n");
					//}
					// call processEdge for all neighbors of vertex in R 
					z = *R.begin();
					R.erase(R.find(z));
				for (edge adj : graph.adjList[z]) {
					int64_t k = adj.to; 
					if (!fixed[k]) {
						processEdge1(graph, z, k);							 
					}
				}
			}	
			
			while (!Q.empty()) {
				for (it=Q.begin(); it!=Q.end(); ++it) {
					int64_t z = *it;
					printf("z in Q:%d\n", z);
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
	// Please note that initialization vector in below format will
	// work fine in C++11, C++14, C++17 but will fail in C++98.
	std::vector<edge> edges;
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
	
	
	// edges.push_back(edge{4,5,4});
	// edges.push_back(edge{4,11,8});
	// edges.push_back(edge{5,6,8});
	// edges.push_back(edge{5,11,11});
	// edges.push_back(edge{6,7,7});
	// edges.push_back(edge{6,12,2});
	// edges.push_back(edge{6,9,4});
	// edges.push_back(edge{7,8,9});
	// edges.push_back(edge{7,9,14});
	// edges.push_back(edge{8,9,10});
	// edges.push_back(edge{9,10,2});
	// edges.push_back(edge{10,11,1});
	// edges.push_back(edge{10,12,6});
	// edges.push_back(edge{11,12,7});

	
	// Maxmum label value of vertices in the given graph, assume 1000
	int64_t N = 15;
	
	// construct graph
	Graph graph(edges, N);

	// print adjacency list representation of graph
	printGraph(graph);

	//Source vertex as first non empty vertex in adjacency List
	int64_t source;
	for(int64_t i = 0; i<nonEmptyIndices.size(); i++) {
			if (nonEmptyIndices[i]) {
				source = i;
			break;
		}
	}
	
	//printf("source:%d\n", source);
	
  //printf("Before Prim\n");
  //fflush( stdout );
	primMST(graph, N, source); 
  //printf("After Prim\n");
  //fflush( stdout );

	std::set<fromTo>::iterator it; //set iterator
	//printf("T size:%d\n", T.size());
	//printf("MST in iterator\n");
	for (it=T.begin(); it!=T.end(); ++it) {
		fromTo e = *it; 
		printf("%d - %d\n", e.from, e.to); 
	}

	return 0; 
} 

//Reference: https://www.geeksforgeeks.org/prims-mst-for-adjacency-list-representation-greedy-algo-6/
// https://www.techiedelight.com/graph-implementation-using-stl/
