

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
	
edges.push_back(edge{134,46,2393});
edges.push_back(edge{134,74,2509});
edges.push_back(edge{134,73,2399});
edges.push_back(edge{106,22,2374});
edges.push_back(edge{106,23,3188});
edges.push_back(edge{106,59,1480});
edges.push_back(edge{106,7,2857});
edges.push_back(edge{106,93,3052});
edges.push_back(edge{106,15,813});
edges.push_back(edge{106,124,577});
edges.push_back(edge{106,84,2249});
edges.push_back(edge{106,46,585});
edges.push_back(edge{106,74,684});
edges.push_back(edge{106,148,666});
edges.push_back(edge{144,146,1445});
edges.push_back(edge{144,54,1192});
edges.push_back(edge{144,135,4411});
edges.push_back(edge{144,102,4411});
edges.push_back(edge{144,59,3665});
edges.push_back(edge{144,101,1453});
edges.push_back(edge{144,71,1763});
edges.push_back(edge{144,92,1967});
edges.push_back(edge{144,124,3595});
edges.push_back(edge{144,57,3989});
edges.push_back(edge{144,141,3597});
edges.push_back(edge{144,84,4038});
edges.push_back(edge{144,47,3691});
edges.push_back(edge{144,73,2641});
edges.push_back(edge{144,139,1784});
edges.push_back(edge{146,54,253});
edges.push_back(edge{146,62,3884});
edges.push_back(edge{146,135,3484});
edges.push_back(edge{146,43,652});
edges.push_back(edge{146,23,4258});
edges.push_back(edge{146,111,1017});
edges.push_back(edge{146,102,3484});
edges.push_back(edge{146,143,2639});
edges.push_back(edge{146,119,413});
edges.push_back(edge{146,35,4038});
edges.push_back(edge{146,59,2714});
edges.push_back(edge{146,117,3362});
edges.push_back(edge{146,7,4103});
edges.push_back(edge{146,101,514});
edges.push_back(edge{146,85,1926});
edges.push_back(edge{146,127,3262});
edges.push_back(edge{146,128,3553});
edges.push_back(edge{146,71,824});
edges.push_back(edge{146,29,4390});
edges.push_back(edge{146,92,1028});
edges.push_back(edge{146,15,2566});
edges.push_back(edge{146,124,2641});
edges.push_back(edge{146,31,4403});
edges.push_back(edge{146,77,4255});
edges.push_back(edge{146,82,4426});
edges.push_back(edge{146,110,3101});
edges.push_back(edge{146,55,2373});
edges.push_back(edge{146,57,3566});
edges.push_back(edge{146,120,792});
edges.push_back(edge{146,141,2658});
edges.push_back(edge{146,84,3169});
edges.push_back(edge{146,50,802});
edges.push_back(edge{146,46,1795});
edges.push_back(edge{146,64,2808});
edges.push_back(edge{146,129,931});
edges.push_back(edge{146,147,2711});
edges.push_back(edge{146,47,2750});
edges.push_back(edge{146,90,879});
edges.push_back(edge{146,73,1714});
edges.push_back(edge{146,70,2353});
edges.push_back(edge{146,139,866});
edges.push_back(edge{146,89,2395});
edges.push_back(edge{146,114,2307});
edges.push_back(edge{146,14,2680});
edges.push_back(edge{146,107,3412});
edges.push_back(edge{146,19,2918});
edges.push_back(edge{146,44,961});
edges.push_back(edge{61,54,289});
edges.push_back(edge{61,143,2675});
edges.push_back(edge{61,35,4067});
edges.push_back(edge{61,59,2694});
edges.push_back(edge{61,117,3398});
edges.push_back(edge{61,101,550});
edges.push_back(edge{61,85,1962});
edges.push_back(edge{61,127,3066});
edges.push_back(edge{61,71,860});
edges.push_back(edge{61,29,4426});
edges.push_back(edge{61,92,1064});
edges.push_back(edge{61,15,2601});
edges.push_back(edge{61,124,2643});
edges.push_back(edge{61,110,3137});
edges.push_back(edge{61,55,2409});
edges.push_back(edge{61,120,828});
edges.push_back(edge{61,84,3135});
edges.push_back(edge{61,50,838});
edges.push_back(edge{61,46,1831});
edges.push_back(edge{61,64,2840});
edges.push_back(edge{61,129,1046});
edges.push_back(edge{61,47,2782});
edges.push_back(edge{61,73,1738});
edges.push_back(edge{61,70,2307});
edges.push_back(edge{61,89,2427});
edges.push_back(edge{61,114,2343});
edges.push_back(edge{61,19,2954});
edges.push_back(edge{21,7,2613});
edges.push_back(edge{21,124,1166});
edges.push_back(edge{21,84,1933});
edges.push_back(edge{21,70,891});
edges.push_back(edge{21,19,1428});
edges.push_back(edge{67,73,1973});
edges.push_back(edge{54,62,3646});
edges.push_back(edge{54,135,3247});
edges.push_back(edge{54,43,399});
edges.push_back(edge{54,22,3516});
edges.push_back(edge{54,45,1886});
edges.push_back(edge{54,23,4036});
edges.push_back(edge{54,111,736});
edges.push_back(edge{54,102,3247});
edges.push_back(edge{54,143,2386});
edges.push_back(edge{54,16,1992});
edges.push_back(edge{54,119,160});
edges.push_back(edge{54,86,3470});
edges.push_back(edge{54,140,3889});
edges.push_back(edge{54,35,3883});
edges.push_back(edge{54,59,2457});
edges.push_back(edge{54,117,3116});
edges.push_back(edge{54,87,329});
edges.push_back(edge{54,7,3864});
edges.push_back(edge{54,101,261});
edges.push_back(edge{54,149,2173});
edges.push_back(edge{54,85,1673});
edges.push_back(edge{54,130,612});
edges.push_back(edge{54,5,1940});
edges.push_back(edge{54,127,3077});
edges.push_back(edge{54,128,3298});
edges.push_back(edge{54,25,3188});
edges.push_back(edge{54,71,573});
edges.push_back(edge{54,29,4158});
edges.push_back(edge{54,92,894});
edges.push_back(edge{54,15,2313});
edges.push_back(edge{54,124,2382});
edges.push_back(edge{54,9,3579});
edges.push_back(edge{54,31,4150});
edges.push_back(edge{54,145,3249});
edges.push_back(edge{54,77,4011});
edges.push_back(edge{54,82,4182});
edges.push_back(edge{54,88,2670});
edges.push_back(edge{54,91,1838});
edges.push_back(edge{54,110,2850});
edges.push_back(edge{54,55,2120});
edges.push_back(edge{54,57,3281});
edges.push_back(edge{54,120,539});
edges.push_back(edge{54,141,2405});
edges.push_back(edge{54,84,2908});
edges.push_back(edge{54,50,553});
edges.push_back(edge{54,46,1545});
edges.push_back(edge{54,150,3524});
edges.push_back(edge{54,64,2555});
edges.push_back(edge{54,20,3986});
edges.push_back(edge{54,129,681});
edges.push_back(edge{54,147,2440});
edges.push_back(edge{54,74,1677});
edges.push_back(edge{54,3,2013});
edges.push_back(edge{54,47,2499});
edges.push_back(edge{54,148,2587});
edges.push_back(edge{54,90,626});
edges.push_back(edge{54,73,1480});
edges.push_back(edge{54,70,2092});
edges.push_back(edge{54,139,602});
edges.push_back(edge{54,89,2146});
edges.push_back(edge{54,80,2138});
edges.push_back(edge{54,114,2060});
edges.push_back(edge{54,14,2423});
edges.push_back(edge{54,107,3159});
edges.push_back(edge{54,39,3355});
edges.push_back(edge{54,19,2674});
edges.push_back(edge{54,44,715});
edges.push_back(edge{54,34,467});
edges.push_back(edge{62,43,4030});
edges.push_back(edge{62,45,2516});
edges.push_back(edge{62,111,4375});
edges.push_back(edge{62,52,2416});
edges.push_back(edge{62,101,3760});
edges.push_back(edge{62,85,2407});
edges.push_back(edge{62,127,4859});
edges.push_back(edge{62,71,3119});
edges.push_back(edge{62,29,4870});
edges.push_back(edge{62,92,2862});
edges.push_back(edge{62,124,3136});
edges.push_back(edge{62,57,4822});
edges.push_back(edge{62,120,4170});
edges.push_back(edge{62,141,3138});
edges.push_back(edge{62,46,2272});
edges.push_back(edge{62,129,3091});
edges.push_back(edge{62,147,3169});
edges.push_back(edge{62,47,3232});
edges.push_back(edge{62,73,2186});
edges.push_back(edge{62,139,3165});
edges.push_back(edge{62,114,2787});
edges.push_back(edge{62,19,3398});
edges.push_back(edge{62,44,3040});
edges.push_back(edge{62,34,3274});
edges.push_back(edge{135,106,1819});
edges.push_back(edge{135,43,3819});
edges.push_back(edge{135,45,2000});
edges.push_back(edge{135,111,3944});
edges.push_back(edge{135,119,3286});
edges.push_back(edge{135,52,1943});
edges.push_back(edge{135,101,3365});
edges.push_back(edge{135,136,2341});
edges.push_back(edge{135,85,2011});
edges.push_back(edge{135,130,2709});
edges.push_back(edge{135,127,4892});
edges.push_back(edge{135,71,2871});
edges.push_back(edge{135,29,4460});
edges.push_back(edge{135,92,2673});
edges.push_back(edge{135,15,2636});
edges.push_back(edge{135,124,2081});
edges.push_back(edge{135,31,4471});
edges.push_back(edge{135,132,2122});
edges.push_back(edge{135,55,2441});
edges.push_back(edge{135,57,4465});
edges.push_back(edge{135,120,3758});
edges.push_back(edge{135,141,3129});
edges.push_back(edge{135,50,3768});
edges.push_back(edge{135,46,1720});
edges.push_back(edge{135,129,2557});
edges.push_back(edge{135,74,1998});
edges.push_back(edge{135,47,2820});
edges.push_back(edge{135,73,1837});
edges.push_back(edge{135,70,2449});
edges.push_back(edge{135,139,2955});
edges.push_back(edge{135,89,1479});
edges.push_back(edge{135,114,1705});
edges.push_back(edge{135,44,2727});
edges.push_back(edge{95,85,2576});
edges.push_back(edge{95,132,2592});
edges.push_back(edge{95,46,2425});
edges.push_back(edge{95,73,2352});
edges.push_back(edge{12,46,2361});
edges.push_back(edge{12,73,2341});
edges.push_back(edge{6,54,3953});
edges.push_back(edge{6,84,783});
edges.push_back(edge{43,144,1591});
edges.push_back(edge{43,23,4394});
edges.push_back(edge{43,111,1124});
edges.push_back(edge{43,102,3829});
edges.push_back(edge{43,143,2785});
edges.push_back(edge{43,119,559});
edges.push_back(edge{43,35,4177});
edges.push_back(edge{43,59,2842});
edges.push_back(edge{43,117,3508});
edges.push_back(edge{43,7,4249});
edges.push_back(edge{43,101,660});
edges.push_back(edge{43,85,2072});
edges.push_back(edge{43,127,3227});
edges.push_back(edge{43,128,3684});
edges.push_back(edge{43,71,970});
edges.push_back(edge{43,29,4542});
edges.push_back(edge{43,92,1343});
edges.push_back(edge{43,15,2711});
edges.push_back(edge{43,124,2773});
edges.push_back(edge{43,31,4549});
edges.push_back(edge{43,77,4401});
edges.push_back(edge{43,82,4572});
edges.push_back(edge{43,110,3275});
edges.push_back(edge{43,57,4229});
edges.push_back(edge{43,141,2804});
edges.push_back(edge{43,84,3322});
edges.push_back(edge{43,50,948});
edges.push_back(edge{43,46,1941});
edges.push_back(edge{43,64,2953});
edges.push_back(edge{43,129,1077});
edges.push_back(edge{43,147,2835});
edges.push_back(edge{43,47,2898});
edges.push_back(edge{43,73,1848});
edges.push_back(edge{43,70,2453});
edges.push_back(edge{43,139,998});
edges.push_back(edge{43,89,2537});
edges.push_back(edge{43,114,2453});
edges.push_back(edge{43,14,2826});
edges.push_back(edge{43,19,3064});
edges.push_back(edge{43,44,1107});
edges.push_back(edge{43,34,771});
edges.push_back(edge{121,46,2306});
edges.push_back(edge{121,73,2316});
edges.push_back(edge{60,85,910});
edges.push_back(edge{60,70,1343});
edges.push_back(edge{60,89,1360});
edges.push_back(edge{60,114,1269});
edges.push_back(edge{56,12,2437});
edges.push_back(edge{56,26,2437});
edges.push_back(edge{56,59,1118});
edges.push_back(edge{56,7,2495});
edges.push_back(edge{56,85,318});
edges.push_back(edge{56,108,2254});
edges.push_back(edge{56,128,1968});
edges.push_back(edge{56,15,954});
edges.push_back(edge{56,124,1048});
edges.push_back(edge{56,77,4373});
edges.push_back(edge{56,142,2409});
edges.push_back(edge{56,30,1954});
edges.push_back(edge{56,84,1815});
edges.push_back(edge{56,64,1201});
edges.push_back(edge{56,47,1144});
edges.push_back(edge{56,14,1072});
edges.push_back(edge{56,107,1804});
edges.push_back(edge{56,19,1310});
edges.push_back(edge{22,124,1797});
edges.push_back(edge{22,46,2148});
edges.push_back(edge{22,73,2396});
edges.push_back(edge{45,143,1248});
edges.push_back(edge{45,36,1740});
edges.push_back(edge{45,52,633});
edges.push_back(edge{45,136,926});
edges.push_back(edge{45,116,1067});
edges.push_back(edge{45,127,3076});
edges.push_back(edge{45,29,3087});
edges.push_back(edge{45,124,810});
edges.push_back(edge{45,55,887});
edges.push_back(edge{45,57,2958});
edges.push_back(edge{45,141,1182});
edges.push_back(edge{45,47,783});
edges.push_back(edge{45,148,747});
edges.push_back(edge{45,73,423});
edges.push_back(edge{45,70,894});
edges.push_back(edge{45,89,523});
edges.push_back(edge{45,114,437});
edges.push_back(edge{45,14,1203});
edges.push_back(edge{23,111,4707});
edges.push_back(edge{23,119,4059});
edges.push_back(edge{23,59,3520});
edges.push_back(edge{23,87,4408});
edges.push_back(edge{23,52,2730});
edges.push_back(edge{23,101,4125});
edges.push_back(edge{23,85,2720});
edges.push_back(edge{23,130,3435});
edges.push_back(edge{23,127,5206});
edges.push_back(edge{23,71,3441});
edges.push_back(edge{23,29,5184});
edges.push_back(edge{23,92,3176});
edges.push_back(edge{23,15,3625});
edges.push_back(edge{23,124,2995});
edges.push_back(edge{23,31,5197});
edges.push_back(edge{23,91,2885});
edges.push_back(edge{23,55,3167});
edges.push_back(edge{23,57,5129});
edges.push_back(edge{23,120,4526});
edges.push_back(edge{23,141,3452});
edges.push_back(edge{23,50,4601});
edges.push_back(edge{23,46,2537});
edges.push_back(edge{23,64,3565});
edges.push_back(edge{23,129,3436});
edges.push_back(edge{23,147,3483});
edges.push_back(edge{23,74,2653});
edges.push_back(edge{23,47,3339});
edges.push_back(edge{23,90,4705});
edges.push_back(edge{23,73,2504});
edges.push_back(edge{23,70,2983});
edges.push_back(edge{23,139,3365});
edges.push_back(edge{23,89,3192});
edges.push_back(edge{23,114,3101});
edges.push_back(edge{23,14,3454});
edges.push_back(edge{23,19,3712});
edges.push_back(edge{23,44,3328});
edges.push_back(edge{23,34,3588});
edges.push_back(edge{111,102,3982});
edges.push_back(edge{111,143,3065});
edges.push_back(edge{111,119,885});
edges.push_back(edge{111,35,4457});
edges.push_back(edge{111,59,3141});
edges.push_back(edge{111,117,3821});
edges.push_back(edge{111,7,4523});
edges.push_back(edge{111,101,563});
edges.push_back(edge{111,85,2398});
edges.push_back(edge{111,127,3644});
edges.push_back(edge{111,128,3999});
edges.push_back(edge{111,71,1302});
edges.push_back(edge{111,29,4862});
edges.push_back(edge{111,92,1500});
edges.push_back(edge{111,15,3030});
edges.push_back(edge{111,124,3280});
edges.push_back(edge{111,31,4875});
edges.push_back(edge{111,77,4687});
edges.push_back(edge{111,82,4898});
edges.push_back(edge{111,110,3559});
edges.push_back(edge{111,55,2845});
edges.push_back(edge{111,57,3522});
edges.push_back(edge{111,141,3061});
edges.push_back(edge{111,84,3716});
edges.push_back(edge{111,46,2267});
edges.push_back(edge{111,64,3266});
edges.push_back(edge{111,147,3161});
edges.push_back(edge{111,47,3199});
edges.push_back(edge{111,73,2158});
edges.push_back(edge{111,70,2812});
edges.push_back(edge{111,139,1331});
edges.push_back(edge{111,89,2824});
edges.push_back(edge{111,114,2779});
edges.push_back(edge{111,14,3048});
edges.push_back(edge{111,19,3344});
edges.push_back(edge{102,106,1819});
edges.push_back(edge{102,45,2000});
edges.push_back(edge{102,119,3286});
edges.push_back(edge{102,52,1943});
edges.push_back(edge{102,101,3365});
edges.push_back(edge{102,136,2341});
edges.push_back(edge{102,85,2011});
edges.push_back(edge{102,130,2709});
edges.push_back(edge{102,127,4892});
edges.push_back(edge{102,71,2871});
edges.push_back(edge{102,29,4460});
edges.push_back(edge{102,92,2673});
edges.push_back(edge{102,15,2636});
edges.push_back(edge{102,124,2081});
edges.push_back(edge{102,31,4471});
edges.push_back(edge{102,132,2122});
edges.push_back(edge{102,55,2441});
edges.push_back(edge{102,57,4465});
edges.push_back(edge{102,120,3758});
edges.push_back(edge{102,141,3129});
edges.push_back(edge{102,50,3768});
edges.push_back(edge{102,46,1720});
edges.push_back(edge{102,129,2557});
edges.push_back(edge{102,74,1998});
edges.push_back(edge{102,47,2820});
edges.push_back(edge{102,73,1837});
edges.push_back(edge{102,70,2449});
edges.push_back(edge{102,139,2955});
edges.push_back(edge{102,89,1479});
edges.push_back(edge{102,114,1705});
edges.push_back(edge{102,44,2727});
edges.push_back(edge{113,73,2043});
edges.push_back(edge{75,46,2212});
edges.push_back(edge{75,73,2210});
edges.push_back(edge{143,144,3578});
edges.push_back(edge{143,135,2707});
edges.push_back(edge{143,102,2707});
edges.push_back(edge{143,35,3266});
edges.push_back(edge{143,117,2597});
edges.push_back(edge{143,52,986});
edges.push_back(edge{143,7,3338});
edges.push_back(edge{143,101,2565});
edges.push_back(edge{143,85,1159});
edges.push_back(edge{143,127,3614});
edges.push_back(edge{143,128,2811});
edges.push_back(edge{143,71,1847});
edges.push_back(edge{143,92,1617});
edges.push_back(edge{143,31,3638});
edges.push_back(edge{143,77,3490});
edges.push_back(edge{143,82,3661});
edges.push_back(edge{143,110,2336});
edges.push_back(edge{143,57,3577});
edges.push_back(edge{143,120,2935});
edges.push_back(edge{143,84,2658});
edges.push_back(edge{143,50,2935});
edges.push_back(edge{143,46,817});
edges.push_back(edge{143,129,1969});
edges.push_back(edge{143,74,933});
edges.push_back(edge{143,73,937});
edges.push_back(edge{143,139,1804});
edges.push_back(edge{143,44,1693});
edges.push_back(edge{143,34,2758});
edges.push_back(edge{26,46,2361});
edges.push_back(edge{26,73,2341});
edges.push_back(edge{16,135,2313});
edges.push_back(edge{16,102,2313});
edges.push_back(edge{16,143,1480});
edges.push_back(edge{16,136,1291});
edges.push_back(edge{16,124,1497});
edges.push_back(edge{16,64,1650});
edges.push_back(edge{16,47,1593});
edges.push_back(edge{16,73,543});
edges.push_back(edge{16,70,1222});
edges.push_back(edge{16,89,1239});
edges.push_back(edge{16,114,1148});
edges.push_back(edge{119,61,449});
edges.push_back(edge{119,62,3475});
edges.push_back(edge{119,35,4078});
edges.push_back(edge{119,59,2622});
edges.push_back(edge{119,117,3269});
edges.push_back(edge{119,7,3694});
edges.push_back(edge{119,101,421});
edges.push_back(edge{119,85,1833});
edges.push_back(edge{119,127,3066});
edges.push_back(edge{119,128,3426});
edges.push_back(edge{119,71,411});
edges.push_back(edge{119,29,4297});
edges.push_back(edge{119,92,645});
edges.push_back(edge{119,15,2475});
edges.push_back(edge{119,124,2407});
edges.push_back(edge{119,31,4310});
edges.push_back(edge{119,77,4116});
edges.push_back(edge{119,82,4333});
edges.push_back(edge{119,110,2771});
edges.push_back(edge{119,57,3280});
edges.push_back(edge{119,120,699});
edges.push_back(edge{119,141,2565});
edges.push_back(edge{119,84,3008});
edges.push_back(edge{119,46,1702});
edges.push_back(edge{119,64,2400});
edges.push_back(edge{119,129,534});
edges.push_back(edge{119,47,2580});
edges.push_back(edge{119,73,1450});
edges.push_back(edge{119,70,2181});
edges.push_back(edge{119,139,536});
edges.push_back(edge{119,89,2068});
edges.push_back(edge{119,114,2108});
edges.push_back(edge{119,14,2460});
edges.push_back(edge{119,19,2719});
edges.push_back(edge{119,44,868});
edges.push_back(edge{119,34,262});
edges.push_back(edge{131,46,2431});
edges.push_back(edge{86,85,2245});
edges.push_back(edge{86,132,2389});
edges.push_back(edge{86,46,2142});
edges.push_back(edge{86,74,2171});
edges.push_back(edge{86,73,2037});
edges.push_back(edge{140,132,2672});
edges.push_back(edge{140,46,2393});
edges.push_back(edge{140,73,2409});
edges.push_back(edge{27,73,2017});
edges.push_back(edge{105,85,2361});
edges.push_back(edge{36,46,2193});
edges.push_back(edge{36,73,1964});
edges.push_back(edge{118,73,1961});
edges.push_back(edge{35,45,2728});
edges.push_back(edge{35,59,3353});
edges.push_back(edge{35,87,4107});
edges.push_back(edge{35,52,2563});
edges.push_back(edge{35,101,4009});
edges.push_back(edge{35,85,2559});
edges.push_back(edge{35,130,3268});
edges.push_back(edge{35,127,5038});
edges.push_back(edge{35,71,3300});
edges.push_back(edge{35,29,5017});
edges.push_back(edge{35,92,3045});
edges.push_back(edge{35,15,3353});
edges.push_back(edge{35,124,2316});
edges.push_back(edge{35,31,5030});
edges.push_back(edge{35,132,2681});
edges.push_back(edge{35,55,3000});
edges.push_back(edge{35,57,4969});
edges.push_back(edge{35,120,4395});
edges.push_back(edge{35,141,3285});
edges.push_back(edge{35,50,4327});
edges.push_back(edge{35,46,3120});
edges.push_back(edge{35,64,3904});
edges.push_back(edge{35,17,2501});
edges.push_back(edge{35,129,3389});
edges.push_back(edge{35,147,3316});
edges.push_back(edge{35,3,2701});
edges.push_back(edge{35,47,3379});
edges.push_back(edge{35,148,2400});
edges.push_back(edge{35,73,2419});
edges.push_back(edge{35,70,3169});
edges.push_back(edge{35,139,3228});
edges.push_back(edge{35,89,2900});
edges.push_back(edge{35,114,2993});
edges.push_back(edge{35,14,3307});
edges.push_back(edge{35,44,3085});
edges.push_back(edge{35,34,3421});
edges.push_back(edge{59,21,1236});
edges.push_back(edge{59,87,2764});
edges.push_back(edge{59,52,1258});
edges.push_back(edge{59,101,2618});
edges.push_back(edge{59,85,1248});
edges.push_back(edge{59,130,1963});
edges.push_back(edge{59,115,1536});
edges.push_back(edge{59,127,3701});
edges.push_back(edge{59,71,1949});
edges.push_back(edge{59,29,3712});
edges.push_back(edge{59,92,1704});
edges.push_back(edge{59,124,904});
edges.push_back(edge{59,31,3725});
edges.push_back(edge{59,132,1376});
edges.push_back(edge{59,55,1695});
edges.push_back(edge{59,57,3654});
edges.push_back(edge{59,120,3019});
edges.push_back(edge{59,50,3011});
edges.push_back(edge{59,46,993});
edges.push_back(edge{59,17,1196});
edges.push_back(edge{59,129,2031});
edges.push_back(edge{59,79,1274});
edges.push_back(edge{59,76,1503});
edges.push_back(edge{59,74,1237});
edges.push_back(edge{59,90,3105});
edges.push_back(edge{59,73,1025});
edges.push_back(edge{59,70,1691});
edges.push_back(edge{59,139,1895});
edges.push_back(edge{59,89,1720});
edges.push_back(edge{59,114,1629});
edges.push_back(edge{59,44,1933});
edges.push_back(edge{59,34,2300});
edges.push_back(edge{117,45,2059});
edges.push_back(edge{117,87,3438});
edges.push_back(edge{117,52,1859});
edges.push_back(edge{117,101,3234});
edges.push_back(edge{117,85,1898});
edges.push_back(edge{117,130,2610});
edges.push_back(edge{117,127,4337});
edges.push_back(edge{117,71,2571});
edges.push_back(edge{117,29,4348});
edges.push_back(edge{117,92,2388});
edges.push_back(edge{117,15,2526});
edges.push_back(edge{117,124,1924});
edges.push_back(edge{117,31,4361});
edges.push_back(edge{117,132,1938});
edges.push_back(edge{117,55,2331});
edges.push_back(edge{117,57,4300});
edges.push_back(edge{117,120,3651});
edges.push_back(edge{117,50,3658});
edges.push_back(edge{117,46,1754});
edges.push_back(edge{117,17,1832});
edges.push_back(edge{117,129,2594});
edges.push_back(edge{117,147,2647});
edges.push_back(edge{117,76,1736});
edges.push_back(edge{117,74,1888});
edges.push_back(edge{117,47,2710});
edges.push_back(edge{117,73,1665});
edges.push_back(edge{117,70,1824});
edges.push_back(edge{117,139,2580});
edges.push_back(edge{117,89,2356});
edges.push_back(edge{117,114,2265});
edges.push_back(edge{117,14,2638});
edges.push_back(edge{117,44,2607});
edges.push_back(edge{117,34,2997});
edges.push_back(edge{87,62,3960});
edges.push_back(edge{87,7,4179});
edges.push_back(edge{87,101,590});
edges.push_back(edge{87,71,900});
edges.push_back(edge{87,92,2458});
edges.push_back(edge{87,15,2644});
edges.push_back(edge{87,124,2674});
edges.push_back(edge{87,110,3177});
edges.push_back(edge{87,84,3175});
edges.push_back(edge{87,50,878});
edges.push_back(edge{87,46,1871});
edges.push_back(edge{87,64,2885});
edges.push_back(edge{87,73,1778});
edges.push_back(edge{87,70,2347});
edges.push_back(edge{87,89,2474});
edges.push_back(edge{87,114,2383});
edges.push_back(edge{2,46,2038});
edges.push_back(edge{2,73,2020});
edges.push_back(edge{38,52,2190});
edges.push_back(edge{38,85,2151});
edges.push_back(edge{38,141,3651});
edges.push_back(edge{38,46,1959});
edges.push_back(edge{38,89,2623});
edges.push_back(edge{48,73,2241});
edges.push_back(edge{13,35,2428});
edges.push_back(edge{13,7,2500});
edges.push_back(edge{13,128,1973});
edges.push_back(edge{13,93,2520});
edges.push_back(edge{13,64,1206});
edges.push_back(edge{1,73,2211});
edges.push_back(edge{151,59,2533});
edges.push_back(edge{151,46,1602});
edges.push_back(edge{151,73,1509});
edges.push_back(edge{151,70,2078});
edges.push_back(edge{52,16,777});
edges.push_back(edge{52,86,2255});
edges.push_back(edge{52,27,2251});
edges.push_back(edge{52,36,2198});
edges.push_back(edge{52,7,2635});
edges.push_back(edge{52,115,746});
edges.push_back(edge{52,127,2911});
edges.push_back(edge{52,128,2108});
edges.push_back(edge{52,25,2132});
edges.push_back(edge{52,93,2655});
edges.push_back(edge{52,15,868});
edges.push_back(edge{52,24,2022});
edges.push_back(edge{52,81,2064});
edges.push_back(edge{52,110,1633});
edges.push_back(edge{52,125,2069});
edges.push_back(edge{52,141,1190});
edges.push_back(edge{52,84,1955});
edges.push_back(edge{52,46,106});
edges.push_back(edge{52,64,1160});
edges.push_back(edge{52,147,979});
edges.push_back(edge{52,47,1039});
edges.push_back(edge{52,73,234});
edges.push_back(edge{52,114,585});
edges.push_back(edge{52,107,1944});
edges.push_back(edge{7,59,3425});
edges.push_back(edge{7,101,4033});
edges.push_back(edge{7,85,2625});
edges.push_back(edge{7,130,3340});
edges.push_back(edge{7,127,5078});
edges.push_back(edge{7,71,3314});
edges.push_back(edge{7,29,5089});
edges.push_back(edge{7,92,3081});
edges.push_back(edge{7,15,3267});
edges.push_back(edge{7,124,3355});
edges.push_back(edge{7,31,5102});
edges.push_back(edge{7,55,3072});
edges.push_back(edge{7,57,5041});
edges.push_back(edge{7,120,4389});
edges.push_back(edge{7,141,3357});
edges.push_back(edge{7,50,4399});
edges.push_back(edge{7,46,2521});
edges.push_back(edge{7,64,3508});
edges.push_back(edge{7,129,3363});
edges.push_back(edge{7,147,3388});
edges.push_back(edge{7,47,3451});
edges.push_back(edge{7,73,2403});
edges.push_back(edge{7,70,3080});
edges.push_back(edge{7,139,3272});
edges.push_back(edge{7,89,3097});
edges.push_back(edge{7,114,3006});
edges.push_back(edge{7,14,3379});
edges.push_back(edge{7,44,3157});
edges.push_back(edge{7,34,4222});
edges.push_back(edge{101,86,3423});
edges.push_back(edge{101,85,1823});
edges.push_back(edge{101,130,803});
edges.push_back(edge{101,5,2024});
edges.push_back(edge{101,127,3333});
edges.push_back(edge{101,128,3426});
edges.push_back(edge{101,71,767});
edges.push_back(edge{101,29,4257});
edges.push_back(edge{101,92,998});
edges.push_back(edge{101,15,2449});
edges.push_back(edge{101,124,2542});
edges.push_back(edge{101,40,2675});
edges.push_back(edge{101,31,4257});
edges.push_back(edge{101,77,4135});
edges.push_back(edge{101,82,4335});
edges.push_back(edge{101,110,2958});
edges.push_back(edge{101,55,2227});
edges.push_back(edge{101,57,3499});
edges.push_back(edge{101,120,801});
edges.push_back(edge{101,141,2537});
edges.push_back(edge{101,84,3169});
edges.push_back(edge{101,50,816});
edges.push_back(edge{101,46,1813});
edges.push_back(edge{101,64,2687});
edges.push_back(edge{101,129,843});
edges.push_back(edge{101,147,2544});
edges.push_back(edge{101,47,2625});
edges.push_back(edge{101,148,2576});
edges.push_back(edge{101,90,373});
edges.push_back(edge{101,73,1626});
edges.push_back(edge{101,70,2259});
edges.push_back(edge{101,139,781});
edges.push_back(edge{101,89,2277});
edges.push_back(edge{101,114,2178});
edges.push_back(edge{101,14,2558});
edges.push_back(edge{101,107,3269});
edges.push_back(edge{101,19,2786});
edges.push_back(edge{101,44,967});
edges.push_back(edge{101,34,648});
edges.push_back(edge{122,46,1751});
edges.push_back(edge{122,73,2278});
edges.push_back(edge{136,60,1412});
edges.push_back(edge{136,85,972});
edges.push_back(edge{136,124,209});
edges.push_back(edge{136,84,2469});
edges.push_back(edge{136,46,697});
edges.push_back(edge{136,73,748});
edges.push_back(edge{149,143,1316});
edges.push_back(edge{149,124,1333});
edges.push_back(edge{149,84,2100});
edges.push_back(edge{149,10,3106});
edges.push_back(edge{149,70,1058});
edges.push_back(edge{149,89,1075});
edges.push_back(edge{149,114,984});
edges.push_back(edge{116,124,393});
edges.push_back(edge{116,73,1347});
edges.push_back(edge{116,70,730});
edges.push_back(edge{33,52,2841});
edges.push_back(edge{33,132,3270});
edges.push_back(edge{33,46,2902});
edges.push_back(edge{33,74,3037});
edges.push_back(edge{33,73,2985});
edges.push_back(edge{85,106,680});
edges.push_back(edge{85,45,623});
edges.push_back(edge{85,75,2418});
edges.push_back(edge{85,87,2002});
edges.push_back(edge{85,127,2901});
edges.push_back(edge{85,128,2098});
edges.push_back(edge{85,71,1133});
edges.push_back(edge{85,29,2912});
edges.push_back(edge{85,92,934});
edges.push_back(edge{85,15,1090});
edges.push_back(edge{85,124,1175});
edges.push_back(edge{85,40,1189});
edges.push_back(edge{85,31,2925});
edges.push_back(edge{85,77,2777});
edges.push_back(edge{85,132,501});
edges.push_back(edge{85,82,2948});
edges.push_back(edge{85,110,1623});
edges.push_back(edge{85,55,895});
edges.push_back(edge{85,57,2863});
edges.push_back(edge{85,120,2212});
edges.push_back(edge{85,141,1176});
edges.push_back(edge{85,84,1945});
edges.push_back(edge{85,50,2222});
edges.push_back(edge{85,64,1330});
edges.push_back(edge{85,129,1133});
edges.push_back(edge{85,147,1211});
edges.push_back(edge{85,76,703});
edges.push_back(edge{85,74,395});
edges.push_back(edge{85,3,788});
edges.push_back(edge{85,47,1271});
edges.push_back(edge{85,148,1088});
edges.push_back(edge{85,90,2299});
edges.push_back(edge{85,73,224});
edges.push_back(edge{85,70,898});
edges.push_back(edge{85,139,1121});
edges.push_back(edge{85,89,917});
edges.push_back(edge{85,80,546});
edges.push_back(edge{85,114,820});
edges.push_back(edge{85,14,1201});
edges.push_back(edge{85,19,1440});
edges.push_back(edge{85,44,1133});
edges.push_back(edge{85,34,1316});
edges.push_back(edge{108,73,2160});
edges.push_back(edge{53,46,1913});
edges.push_back(edge{53,73,1895});
edges.push_back(edge{49,73,1858});
edges.push_back(edge{68,46,2478});
edges.push_back(edge{130,85,1174});
edges.push_back(edge{130,128,2824});
edges.push_back(edge{130,71,41});
edges.push_back(edge{130,29,4669});
edges.push_back(edge{130,92,275});
edges.push_back(edge{130,15,1816});
edges.push_back(edge{130,124,1904});
edges.push_back(edge{130,110,2349});
edges.push_back(edge{130,55,1621});
edges.push_back(edge{130,141,1906});
edges.push_back(edge{130,64,2057});
edges.push_back(edge{130,47,2000});
edges.push_back(edge{130,73,1023});
edges.push_back(edge{130,70,1629});
edges.push_back(edge{130,19,2166});
edges.push_back(edge{115,114,1117});
edges.push_back(edge{5,124,1445});
edges.push_back(edge{127,56,2716});
edges.push_back(edge{127,16,3220});
edges.push_back(edge{127,100,5919});
edges.push_back(edge{127,128,4551});
edges.push_back(edge{127,71,3505});
edges.push_back(edge{127,92,3425});
edges.push_back(edge{127,15,3727});
edges.push_back(edge{127,124,3617});
edges.push_back(edge{127,77,5230});
edges.push_back(edge{127,132,2859});
edges.push_back(edge{127,82,5401});
edges.push_back(edge{127,91,3066});
edges.push_back(edge{127,110,4212});
edges.push_back(edge{127,55,3348});
edges.push_back(edge{127,120,3690});
edges.push_back(edge{127,141,3633});
edges.push_back(edge{127,84,4450});
edges.push_back(edge{127,50,3326});
edges.push_back(edge{127,46,2608});
edges.push_back(edge{127,64,3753});
edges.push_back(edge{127,17,2849});
edges.push_back(edge{127,129,3582});
edges.push_back(edge{127,147,3664});
edges.push_back(edge{127,74,2812});
edges.push_back(edge{127,47,3729});
edges.push_back(edge{127,148,3585});
edges.push_back(edge{127,90,4583});
edges.push_back(edge{127,73,2689});
edges.push_back(edge{127,70,3395});
edges.push_back(edge{127,139,3481});
edges.push_back(edge{127,89,3453});
edges.push_back(edge{127,114,3282});
edges.push_back(edge{127,14,3655});
edges.push_back(edge{127,107,4387});
edges.push_back(edge{127,44,3459});
edges.push_back(edge{72,59,2590});
edges.push_back(edge{104,46,2477});
edges.push_back(edge{104,73,2459});
edges.push_back(edge{41,52,2429});
edges.push_back(edge{41,46,2323});
edges.push_back(edge{100,124,2288});
edges.push_back(edge{100,3,2678});
edges.push_back(edge{100,73,3242});
edges.push_back(edge{128,71,2794});
edges.push_back(edge{128,29,4562});
edges.push_back(edge{128,92,2554});
edges.push_back(edge{128,15,2740});
edges.push_back(edge{128,124,2828});
edges.push_back(edge{128,31,4575});
edges.push_back(edge{128,55,2545});
edges.push_back(edge{128,120,3847});
edges.push_back(edge{128,141,2830});
edges.push_back(edge{128,50,3815});
edges.push_back(edge{128,46,2003});
edges.push_back(edge{128,129,2808});
edges.push_back(edge{128,47,2924});
edges.push_back(edge{128,90,3911});
edges.push_back(edge{128,73,1888});
edges.push_back(edge{128,70,2553});
edges.push_back(edge{128,139,2807});
edges.push_back(edge{128,89,2570});
edges.push_back(edge{128,114,2479});
edges.push_back(edge{128,34,3638});
edges.push_back(edge{0,47,1616});
edges.push_back(edge{25,46,1916});
edges.push_back(edge{25,73,1898});
edges.push_back(edge{66,73,2523});
edges.push_back(edge{99,54,3984});
edges.push_back(edge{99,46,2442});
edges.push_back(edge{99,73,2580});
edges.push_back(edge{93,54,3870});
edges.push_back(edge{93,45,2280});
edges.push_back(edge{93,85,2645});
edges.push_back(edge{93,124,2475});
edges.push_back(edge{93,132,2828});
edges.push_back(edge{93,46,2843});
edges.push_back(edge{93,73,2421});
edges.push_back(edge{93,70,2586});
edges.push_back(edge{93,89,2783});
edges.push_back(edge{71,99,3528});
edges.push_back(edge{71,29,3761});
edges.push_back(edge{71,92,234});
edges.push_back(edge{71,15,1782});
edges.push_back(edge{71,124,1888});
edges.push_back(edge{71,31,3769});
edges.push_back(edge{71,77,3490});
edges.push_back(edge{71,82,3633});
edges.push_back(edge{71,110,2308});
edges.push_back(edge{71,55,1580});
edges.push_back(edge{71,57,3600});
edges.push_back(edge{71,120,1110});
edges.push_back(edge{71,141,1866});
edges.push_back(edge{71,84,2840});
edges.push_back(edge{71,50,1120});
edges.push_back(edge{71,46,1950});
edges.push_back(edge{71,64,2016});
edges.push_back(edge{71,129,149});
edges.push_back(edge{71,147,1909});
edges.push_back(edge{71,47,1967});
edges.push_back(edge{71,90,1197});
edges.push_back(edge{71,73,930});
edges.push_back(edge{71,70,1610});
edges.push_back(edge{71,139,95});
edges.push_back(edge{71,89,1606});
edges.push_back(edge{71,114,1515});
edges.push_back(edge{71,14,1903});
edges.push_back(edge{71,107,2623});
edges.push_back(edge{71,19,2125});
edges.push_back(edge{71,44,227});
edges.push_back(edge{71,34,199});
edges.push_back(edge{29,52,2922});
edges.push_back(edge{29,92,3368});
edges.push_back(edge{29,15,3523});
edges.push_back(edge{29,124,3514});
edges.push_back(edge{29,77,5241});
edges.push_back(edge{29,132,3039});
edges.push_back(edge{29,82,5412});
edges.push_back(edge{29,110,4087});
edges.push_back(edge{29,55,2380});
edges.push_back(edge{29,120,4676});
edges.push_back(edge{29,141,3644});
edges.push_back(edge{29,84,4404});
edges.push_back(edge{29,50,4686});
edges.push_back(edge{29,46,2616});
edges.push_back(edge{29,64,3775});
edges.push_back(edge{29,129,3597});
edges.push_back(edge{29,147,3740});
edges.push_back(edge{29,76,3167});
edges.push_back(edge{29,74,2916});
edges.push_back(edge{29,47,3678});
edges.push_back(edge{29,73,2696});
edges.push_back(edge{29,70,3464});
edges.push_back(edge{29,139,3683});
edges.push_back(edge{29,89,2387});
edges.push_back(edge{29,114,3293});
edges.push_back(edge{29,14,3616});
edges.push_back(edge{29,19,3904});
edges.push_back(edge{29,44,3444});
edges.push_back(edge{92,16,1223});
edges.push_back(edge{92,2,2700});
edges.push_back(edge{92,15,1546});
edges.push_back(edge{92,124,1634});
edges.push_back(edge{92,31,3381});
edges.push_back(edge{92,77,3233});
edges.push_back(edge{92,82,3419});
edges.push_back(edge{92,110,2079});
edges.push_back(edge{92,55,1351});
edges.push_back(edge{92,57,3333});
edges.push_back(edge{92,120,1317});
edges.push_back(edge{92,141,1636});
edges.push_back(edge{92,84,2430});
edges.push_back(edge{92,50,1775});
edges.push_back(edge{92,64,1787});
edges.push_back(edge{92,129,111});
edges.push_back(edge{92,147,1673});
edges.push_back(edge{92,47,1730});
edges.push_back(edge{92,73,689});
edges.push_back(edge{92,70,1357});
edges.push_back(edge{92,139,226});
edges.push_back(edge{92,89,1376});
edges.push_back(edge{92,114,1285});
edges.push_back(edge{92,14,1658});
edges.push_back(edge{92,107,2390});
edges.push_back(edge{92,19,1896});
edges.push_back(edge{92,44,82});
edges.push_back(edge{92,34,700});
edges.push_back(edge{15,16,1409});
edges.push_back(edge{15,38,2793});
edges.push_back(edge{15,115,1378});
edges.push_back(edge{15,124,236});
edges.push_back(edge{15,31,3567});
edges.push_back(edge{15,77,3419});
edges.push_back(edge{15,132,1061});
edges.push_back(edge{15,82,3590});
edges.push_back(edge{15,91,1255});
edges.push_back(edge{15,110,2265});
edges.push_back(edge{15,57,3496});
edges.push_back(edge{15,120,2853});
edges.push_back(edge{15,84,2550});
edges.push_back(edge{15,50,2864});
edges.push_back(edge{15,46,762});
edges.push_back(edge{15,17,1039});
edges.push_back(edge{15,129,1865});
edges.push_back(edge{15,76,737});
edges.push_back(edge{15,74,913});
edges.push_back(edge{15,90,2941});
edges.push_back(edge{15,73,866});
edges.push_back(edge{15,70,414});
edges.push_back(edge{15,139,1747});
edges.push_back(edge{15,11,454});
edges.push_back(edge{15,44,1753});
edges.push_back(edge{15,34,2687});
edges.push_back(edge{124,140,3233});
edges.push_back(edge{124,36,1900});
edges.push_back(edge{124,38,1979});
edges.push_back(edge{124,52,1007});
edges.push_back(edge{124,31,3655});
edges.push_back(edge{124,77,3507});
edges.push_back(edge{124,132,630});
edges.push_back(edge{124,82,3296});
edges.push_back(edge{124,58,256});
edges.push_back(edge{124,88,267});
edges.push_back(edge{124,91,1343});
edges.push_back(edge{124,110,2272});
edges.push_back(edge{124,57,3490});
edges.push_back(edge{124,120,2909});
edges.push_back(edge{124,84,1973});
edges.push_back(edge{124,50,2913});
edges.push_back(edge{124,46,834});
edges.push_back(edge{124,17,1126});
edges.push_back(edge{124,129,2035});
edges.push_back(edge{124,76,501});
edges.push_back(edge{124,74,908});
edges.push_back(edge{124,112,3220});
edges.push_back(edge{124,3,390});
edges.push_back(edge{124,90,3029});
edges.push_back(edge{124,73,955});
edges.push_back(edge{124,70,337});
edges.push_back(edge{124,139,1865});
edges.push_back(edge{124,89,308});
edges.push_back(edge{124,114,402});
edges.push_back(edge{124,107,1771});
edges.push_back(edge{124,11,399});
edges.push_back(edge{124,137,696});
edges.push_back(edge{124,44,1812});
edges.push_back(edge{124,34,2563});
edges.push_back(edge{65,46,1898});
edges.push_back(edge{65,17,2114});
edges.push_back(edge{103,73,2044});
edges.push_back(edge{9,46,2472});
edges.push_back(edge{9,73,2454});
edges.push_back(edge{40,46,846});
edges.push_back(edge{40,73,965});
edges.push_back(edge{31,60,3365});
edges.push_back(edge{31,45,3100});
edges.push_back(edge{31,16,3244});
edges.push_back(edge{31,52,2935});
edges.push_back(edge{31,136,3449});
edges.push_back(edge{31,77,5254});
edges.push_back(edge{31,132,3052});
edges.push_back(edge{31,82,5425});
edges.push_back(edge{31,110,4100});
edges.push_back(edge{31,55,3372});
edges.push_back(edge{31,120,4689});
edges.push_back(edge{31,141,3657});
edges.push_back(edge{31,84,4422});
edges.push_back(edge{31,50,4699});
edges.push_back(edge{31,46,2830});
edges.push_back(edge{31,64,3808});
edges.push_back(edge{31,129,3651});
edges.push_back(edge{31,147,3744});
edges.push_back(edge{31,74,2929});
edges.push_back(edge{31,3,3265});
edges.push_back(edge{31,47,3751});
edges.push_back(edge{31,73,2704});
edges.push_back(edge{31,70,3380});
edges.push_back(edge{31,139,3586});
edges.push_back(edge{31,89,3364});
edges.push_back(edge{31,114,3306});
edges.push_back(edge{31,14,3679});
edges.push_back(edge{31,44,3457});
edges.push_back(edge{31,34,4522});
edges.push_back(edge{126,46,1920});
edges.push_back(edge{126,73,2008});
edges.push_back(edge{98,126,2534});
edges.push_back(edge{77,61,4291});
edges.push_back(edge{77,59,3577});
edges.push_back(edge{77,52,2787});
edges.push_back(edge{77,55,3224});
edges.push_back(edge{77,57,5193});
edges.push_back(edge{77,120,4541});
edges.push_back(edge{77,141,3509});
edges.push_back(edge{77,50,4551});
edges.push_back(edge{77,46,2682});
edges.push_back(edge{77,64,3660});
edges.push_back(edge{77,17,2725});
edges.push_back(edge{77,129,3473});
edges.push_back(edge{77,147,3540});
edges.push_back(edge{77,47,3603});
edges.push_back(edge{77,90,4628});
edges.push_back(edge{77,73,2554});
edges.push_back(edge{77,70,3232});
edges.push_back(edge{77,139,3415});
edges.push_back(edge{77,89,3249});
edges.push_back(edge{77,114,3158});
edges.push_back(edge{77,14,3531});
edges.push_back(edge{77,19,3769});
edges.push_back(edge{77,44,3309});
edges.push_back(edge{142,46,2333});
edges.push_back(edge{142,74,2543});
edges.push_back(edge{142,73,2315});
edges.push_back(edge{32,46,1999});
edges.push_back(edge{32,89,2246});
edges.push_back(edge{32,114,2475});
edges.push_back(edge{132,45,566});
edges.push_back(edge{132,23,2803});
edges.push_back(edge{132,16,894});
edges.push_back(edge{132,38,2279});
edges.push_back(edge{132,128,2226});
edges.push_back(edge{132,24,2140});
edges.push_back(edge{132,57,2992});
edges.push_back(edge{132,84,2133});
edges.push_back(edge{132,46,222});
edges.push_back(edge{132,64,1382});
edges.push_back(edge{132,148,719});
edges.push_back(edge{132,73,351});
edges.push_back(edge{82,106,2919});
edges.push_back(edge{82,59,4526});
edges.push_back(edge{82,55,3395});
edges.push_back(edge{82,57,5364});
edges.push_back(edge{82,120,4712});
edges.push_back(edge{82,46,2853});
edges.push_back(edge{82,64,3831});
edges.push_back(edge{82,129,3635});
edges.push_back(edge{82,147,3767});
edges.push_back(edge{82,47,3774});
edges.push_back(edge{82,73,2727});
edges.push_back(edge{82,70,2962});
edges.push_back(edge{82,139,3600});
edges.push_back(edge{82,89,3323});
edges.push_back(edge{82,114,3329});
edges.push_back(edge{82,14,3702});
edges.push_back(edge{24,46,1806});
edges.push_back(edge{30,73,1860});
edges.push_back(edge{58,23,2867});
edges.push_back(edge{58,59,1118});
edges.push_back(edge{58,84,2001});
edges.push_back(edge{58,46,582});
edges.push_back(edge{81,46,1848});
edges.push_back(edge{81,73,1830});
edges.push_back(edge{91,29,3077});
edges.push_back(edge{91,70,1068});
edges.push_back(edge{91,89,1085});
edges.push_back(edge{91,114,994});
edges.push_back(edge{110,106,1855});
edges.push_back(edge{110,45,1798});
edges.push_back(edge{110,132,1751});
edges.push_back(edge{110,55,2070});
edges.push_back(edge{110,57,4039});
edges.push_back(edge{110,120,3387});
edges.push_back(edge{110,141,2355});
edges.push_back(edge{110,50,3397});
edges.push_back(edge{110,46,1520});
edges.push_back(edge{110,64,2506});
edges.push_back(edge{110,17,1412});
edges.push_back(edge{110,129,2186});
edges.push_back(edge{110,147,2386});
edges.push_back(edge{110,47,2449});
edges.push_back(edge{110,73,1399});
edges.push_back(edge{110,70,2078});
edges.push_back(edge{110,139,2380});
edges.push_back(edge{110,89,2095});
edges.push_back(edge{110,114,2004});
edges.push_back(edge{110,14,2377});
edges.push_back(edge{110,44,2155});
edges.push_back(edge{110,34,2496});
edges.push_back(edge{123,54,4007});
edges.push_back(edge{123,52,2841});
edges.push_back(edge{123,84,837});
edges.push_back(edge{123,46,2590});
edges.push_back(edge{123,73,2582});
edges.push_back(edge{123,14,3456});
edges.push_back(edge{55,119,2280});
edges.push_back(edge{55,38,2598});
edges.push_back(edge{55,52,777});
edges.push_back(edge{55,115,1183});
edges.push_back(edge{55,81,2501});
edges.push_back(edge{55,57,2367});
edges.push_back(edge{55,120,2659});
edges.push_back(edge{55,84,2392});
edges.push_back(edge{55,46,543});
edges.push_back(edge{55,129,1678});
edges.push_back(edge{55,147,418});
edges.push_back(edge{55,74,899});
edges.push_back(edge{55,90,2746});
edges.push_back(edge{55,73,671});
edges.push_back(edge{55,139,1546});
edges.push_back(edge{55,107,2381});
edges.push_back(edge{55,44,1549});
edges.push_back(edge{55,34,2249});
edges.push_back(edge{57,16,3183});
edges.push_back(edge{57,52,2874});
edges.push_back(edge{57,149,3019});
edges.push_back(edge{57,120,3336});
edges.push_back(edge{57,141,3596});
edges.push_back(edge{57,84,4355});
edges.push_back(edge{57,50,3992});
edges.push_back(edge{57,46,2567});
edges.push_back(edge{57,64,3710});
edges.push_back(edge{57,17,2812});
edges.push_back(edge{57,129,3427});
edges.push_back(edge{57,74,2868});
edges.push_back(edge{57,47,2596});
edges.push_back(edge{57,73,2659});
edges.push_back(edge{57,70,3319});
edges.push_back(edge{57,139,3493});
edges.push_back(edge{57,89,2376});
edges.push_back(edge{57,114,2412});
edges.push_back(edge{57,14,3592});
edges.push_back(edge{57,19,3856});
edges.push_back(edge{57,44,3450});
edges.push_back(edge{57,34,3732});
edges.push_back(edge{125,46,2111});
edges.push_back(edge{125,73,2414});
edges.push_back(edge{37,73,1681});
edges.push_back(edge{120,87,868});
edges.push_back(edge{120,141,2944});
edges.push_back(edge{120,84,3509});
edges.push_back(edge{120,50,183});
edges.push_back(edge{120,46,2081});
edges.push_back(edge{120,64,3094});
edges.push_back(edge{120,129,1217});
edges.push_back(edge{120,147,2975});
edges.push_back(edge{120,47,3034});
edges.push_back(edge{120,73,1988});
edges.push_back(edge{120,70,2648});
edges.push_back(edge{120,139,1139});
edges.push_back(edge{120,89,2684});
edges.push_back(edge{120,114,2593});
edges.push_back(edge{120,14,2966});
edges.push_back(edge{120,19,3204});
edges.push_back(edge{120,44,1247});
edges.push_back(edge{141,60,1620});
edges.push_back(edge{141,117,2616});
edges.push_back(edge{141,87,2734});
edges.push_back(edge{141,115,1468});
edges.push_back(edge{141,72,2522});
edges.push_back(edge{141,132,1060});
edges.push_back(edge{141,84,2677});
edges.push_back(edge{141,50,2954});
edges.push_back(edge{141,46,838});
edges.push_back(edge{141,17,1128});
edges.push_back(edge{141,129,1988});
edges.push_back(edge{141,74,1011});
edges.push_back(edge{141,73,956});
edges.push_back(edge{141,139,1818});
edges.push_back(edge{141,107,2666});
edges.push_back(edge{141,44,1865});
edges.push_back(edge{141,34,2412});
edges.push_back(edge{84,45,2083});
edges.push_back(edge{84,16,2264});
edges.push_back(edge{84,50,3492});
edges.push_back(edge{84,46,1744});
edges.push_back(edge{84,64,2828});
edges.push_back(edge{84,17,1893});
edges.push_back(edge{84,129,2630});
edges.push_back(edge{84,147,2660});
edges.push_back(edge{84,74,1902});
edges.push_back(edge{84,47,2114});
edges.push_back(edge{84,90,3580});
edges.push_back(edge{84,73,1733});
edges.push_back(edge{84,70,2058});
edges.push_back(edge{84,139,2783});
edges.push_back(edge{84,89,2356});
edges.push_back(edge{84,114,2312});
edges.push_back(edge{84,14,2684});
edges.push_back(edge{84,19,2937});
edges.push_back(edge{84,44,3195});
edges.push_back(edge{84,34,3015});
edges.push_back(edge{50,62,4180});
edges.push_back(edge{50,55,2669});
edges.push_back(edge{50,46,2091});
edges.push_back(edge{50,64,3104});
edges.push_back(edge{50,129,1227});
edges.push_back(edge{50,147,2985});
edges.push_back(edge{50,47,3048});
edges.push_back(edge{50,73,1998});
edges.push_back(edge{50,70,2603});
edges.push_back(edge{50,139,1163});
edges.push_back(edge{50,89,2694});
edges.push_back(edge{50,114,2603});
edges.push_back(edge{50,14,2976});
edges.push_back(edge{50,107,3708});
edges.push_back(edge{50,34,921});
edges.push_back(edge{10,46,2744});
edges.push_back(edge{10,76,2992});
edges.push_back(edge{10,73,2787});
edges.push_back(edge{46,60,793});
edges.push_back(edge{46,45,387});
edges.push_back(edge{46,16,672});
edges.push_back(edge{46,118,1979});
edges.push_back(edge{46,1,2167});
edges.push_back(edge{46,116,1227});
edges.push_back(edge{46,94,1896});
edges.push_back(edge{46,85,316});
edges.push_back(edge{46,83,1923});
edges.push_back(edge{46,49,1876});
edges.push_back(edge{46,115,641});
edges.push_back(edge{46,100,3122});
edges.push_back(edge{46,66,2513});
edges.push_back(edge{46,42,2098});
edges.push_back(edge{46,145,2076});
edges.push_back(edge{46,64,1009});
edges.push_back(edge{46,138,2151});
edges.push_back(edge{46,20,2444});
edges.push_back(edge{46,147,874});
edges.push_back(edge{46,76,608});
edges.push_back(edge{46,74,116});
edges.push_back(edge{46,112,2492});
edges.push_back(edge{46,3,958});
edges.push_back(edge{46,47,933});
edges.push_back(edge{46,63,2780});
edges.push_back(edge{46,133,2421});
edges.push_back(edge{46,148,830});
edges.push_back(edge{46,90,2168});
edges.push_back(edge{46,18,2025});
edges.push_back(edge{46,73,129});
edges.push_back(edge{46,70,550});
edges.push_back(edge{46,89,569});
edges.push_back(edge{46,80,630});
edges.push_back(edge{46,114,479});
edges.push_back(edge{46,14,859});
edges.push_back(edge{46,107,2291});
edges.push_back(edge{46,137,440});
edges.push_back(edge{46,78,2665});
edges.push_back(edge{46,39,1853});
edges.push_back(edge{46,19,1345});
edges.push_back(edge{46,97,1963});
edges.push_back(edge{46,51,2324});
edges.push_back(edge{150,85,2696});
edges.push_back(edge{150,46,2528});
edges.push_back(edge{150,73,2399});
edges.push_back(edge{64,106,1563});
edges.push_back(edge{64,91,1496});
edges.push_back(edge{64,17,1287});
edges.push_back(edge{64,129,2101});
edges.push_back(edge{64,74,1230});
edges.push_back(edge{64,90,3182});
edges.push_back(edge{64,73,1108});
edges.push_back(edge{64,139,2007});
edges.push_back(edge{64,89,1578});
edges.push_back(edge{64,114,1712});
edges.push_back(edge{64,44,2057});
edges.push_back(edge{64,34,2497});
edges.push_back(edge{28,73,1851});
edges.push_back(edge{138,52,2367});
edges.push_back(edge{138,132,2485});
edges.push_back(edge{138,74,2267});
edges.push_back(edge{138,73,2133});
edges.push_back(edge{20,3,3018});
edges.push_back(edge{20,73,2454});
edges.push_back(edge{17,135,1942});
edges.push_back(edge{17,102,1942});
edges.push_back(edge{17,143,1109});
edges.push_back(edge{17,128,2046});
edges.push_back(edge{17,46,301});
edges.push_back(edge{17,3,736});
edges.push_back(edge{17,47,1224});
edges.push_back(edge{17,14,1150});
edges.push_back(edge{17,19,1388});
edges.push_back(edge{129,147,1988});
edges.push_back(edge{129,47,2066});
edges.push_back(edge{129,73,951});
edges.push_back(edge{129,70,1692});
edges.push_back(edge{129,89,1693});
edges.push_back(edge{129,114,1582});
edges.push_back(edge{129,14,1846});
edges.push_back(edge{129,107,2742});
edges.push_back(edge{129,19,2125});
edges.push_back(edge{129,44,31});
edges.push_back(edge{147,119,2280});
edges.push_back(edge{147,130,1926});
edges.push_back(edge{147,93,3007});
edges.push_back(edge{147,132,1338});
edges.push_back(edge{147,57,3627});
edges.push_back(edge{147,74,1064});
edges.push_back(edge{147,73,996});
edges.push_back(edge{147,70,421});
edges.push_back(edge{147,139,1890});
edges.push_back(edge{147,89,534});
edges.push_back(edge{147,114,440});
edges.push_back(edge{147,44,1743});
edges.push_back(edge{147,34,2087});
edges.push_back(edge{79,12,2593});
edges.push_back(edge{79,26,2593});
edges.push_back(edge{79,7,2651});
edges.push_back(edge{76,93,2976});
edges.push_back(edge{76,84,2246});
edges.push_back(edge{76,17,651});
edges.push_back(edge{76,47,610});
edges.push_back(edge{76,73,479});
edges.push_back(edge{74,75,2422});
edges.push_back(edge{74,36,2209});
edges.push_back(edge{74,35,3037});
edges.push_back(edge{74,7,2629});
edges.push_back(edge{74,24,1922});
edges.push_back(edge{74,112,2502});
edges.push_back(edge{74,47,1286});
edges.push_back(edge{74,148,815});
edges.push_back(edge{74,73,228});
edges.push_back(edge{74,78,2661});
edges.push_back(edge{112,73,2538});
edges.push_back(edge{3,60,1228});
edges.push_back(edge{3,22,2187});
edges.push_back(edge{3,23,3030});
edges.push_back(edge{3,38,2491});
edges.push_back(edge{3,52,798});
edges.push_back(edge{3,7,2965});
edges.push_back(edge{3,127,3241});
edges.push_back(edge{3,93,2985});
edges.push_back(edge{3,74,796});
edges.push_back(edge{3,47,499});
edges.push_back(edge{3,133,2551});
edges.push_back(edge{3,148,479});
edges.push_back(edge{3,73,564});
edges.push_back(edge{109,46,2687});
edges.push_back(edge{109,73,2720});
edges.push_back(edge{47,106,686});
edges.push_back(edge{47,86,3087});
edges.push_back(edge{47,132,1121});
edges.push_back(edge{47,90,3125});
edges.push_back(edge{47,73,1050});
edges.push_back(edge{47,139,1944});
edges.push_back(edge{47,89,417});
edges.push_back(edge{47,44,2112});
edges.push_back(edge{47,34,2506});
edges.push_back(edge{63,73,2762});
edges.push_back(edge{133,101,3930});
edges.push_back(edge{133,73,2426});
edges.push_back(edge{148,73,923});
edges.push_back(edge{148,114,310});
edges.push_back(edge{90,111,204});
edges.push_back(edge{90,143,3012});
edges.push_back(edge{90,120,1165});
edges.push_back(edge{90,73,2090});
edges.push_back(edge{90,70,2644});
edges.push_back(edge{90,114,2680});
edges.push_back(edge{18,73,2007});
edges.push_back(edge{73,106,456});
edges.push_back(edge{73,60,664});
edges.push_back(edge{73,56,94});
edges.push_back(edge{73,105,2137});
edges.push_back(edge{73,38,2128});
edges.push_back(edge{73,13,99});
edges.push_back(edge{73,83,1888});
edges.push_back(edge{73,68,2607});
edges.push_back(edge{73,42,1969});
edges.push_back(edge{73,145,2120});
edges.push_back(edge{73,32,1984});
edges.push_back(edge{73,24,1788});
edges.push_back(edge{73,88,1221});
edges.push_back(edge{73,79,250});
edges.push_back(edge{73,4,1889});
edges.push_back(edge{73,70,679});
edges.push_back(edge{73,139,923});
edges.push_back(edge{73,89,696});
edges.push_back(edge{73,96,3769});
edges.push_back(edge{73,114,605});
edges.push_back(edge{73,14,978});
edges.push_back(edge{73,107,1717});
edges.push_back(edge{73,137,474});
edges.push_back(edge{73,78,2649});
edges.push_back(edge{73,39,1942});
edges.push_back(edge{73,19,1216});
edges.push_back(edge{73,51,2334});
edges.push_back(edge{73,44,795});
edges.push_back(edge{73,34,1157});
edges.push_back(edge{73,8,103});
edges.push_back(edge{70,62,2861});
edges.push_back(edge{70,52,784});
edges.push_back(edge{70,115,1191});
edges.push_back(edge{70,17,851});
edges.push_back(edge{70,74,762});
edges.push_back(edge{70,112,3316});
edges.push_back(edge{70,139,1554});
edges.push_back(edge{70,107,1736});
edges.push_back(edge{70,44,1435});
edges.push_back(edge{70,34,1771});
edges.push_back(edge{139,99,3433});
edges.push_back(edge{139,42,2973});
edges.push_back(edge{139,46,2134});
edges.push_back(edge{139,89,1574});
edges.push_back(edge{139,114,1470});
edges.push_back(edge{139,14,1860});
edges.push_back(edge{139,107,2714});
edges.push_back(edge{139,19,2120});
edges.push_back(edge{139,44,248});
edges.push_back(edge{139,34,294});
edges.push_back(edge{89,62,2670});
edges.push_back(edge{89,52,802});
edges.push_back(edge{89,74,685});
edges.push_back(edge{89,148,404});
edges.push_back(edge{89,107,2237});
edges.push_back(edge{89,44,1605});
edges.push_back(edge{89,34,2152});
edges.push_back(edge{80,106,1152});
edges.push_back(edge{80,52,736});
edges.push_back(edge{80,132,1041});
edges.push_back(edge{80,73,1544});
edges.push_back(edge{114,130,1544});
edges.push_back(edge{114,132,701});
edges.push_back(edge{114,74,595});
edges.push_back(edge{114,107,2315});
edges.push_back(edge{114,44,1552});
edges.push_back(edge{114,34,2217});
edges.push_back(edge{14,61,2716});
edges.push_back(edge{14,52,1057});
edges.push_back(edge{14,130,1922});
edges.push_back(edge{14,132,1247});
edges.push_back(edge{14,74,1206});
edges.push_back(edge{14,107,2688});
edges.push_back(edge{14,44,1734});
edges.push_back(edge{107,85,2101});
edges.push_back(edge{107,120,3698});
edges.push_back(edge{107,76,2189});
edges.push_back(edge{107,74,2207});
edges.push_back(edge{137,64,1581});
edges.push_back(edge{137,148,785});
edges.push_back(edge{78,52,2823});
edges.push_back(edge{78,85,2957});
edges.push_back(edge{78,132,2941});
edges.push_back(edge{39,74,2125});
edges.push_back(edge{19,127,3893});
edges.push_back(edge{19,31,3917});
edges.push_back(edge{19,70,1895});
edges.push_back(edge{19,89,1912});
edges.push_back(edge{19,114,1821});
edges.push_back(edge{69,59,2599});
edges.push_back(edge{69,73,1575});
edges.push_back(edge{44,144,1900});
edges.push_back(edge{44,128,2630});
edges.push_back(edge{44,82,3480});
edges.push_back(edge{34,146,625});
edges.push_back(edge{34,77,3645});
edges.push_back(edge{34,46,2046});
edges.push_back(edge{34,129,322});
edges.push_back(edge{34,14,2070});
edges.push_back(edge{8,59,1127});
edges.push_back(edge{8,7,2504});
edges.push_back(edge{8,128,1977});
edges.push_back(edge{8,84,1824});

	
	
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