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
#include <limits.h>       //For PATH_MAX
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

//Identifies all minimum weight edges for all vertices
void initMWE(Graph const &graph) 
{ 
	for (int i = 0; i < graph.adjList.size(); i++) {
		// Extract the vertex with minimum dist value 
		int prevWeight=INT_MAX;
		int min_to, minFrom;
		// Iterate through all the vertices of graph 
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
    printf("Process Edge from %i, to %i\n", z, k);
	int weight = getWeight(graph, z, k);
	if (mwe.find(fromTo{z, k}) != mwe.end()) {
		fixed[k] = true;
		T.insert(fromTo{k, z}); // z is the parent of k
		R.insert(k);
	}
	else if (dist[k] > weight) {
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
std::set<fromTo> primMST(Graph const &graph, int N, int source) 
{
	// Initialize min heap with all vertices. dist value of 
	// all vertices (except 0th vertex) is initially infinite 
//    printf("primMST\n");
	for(int i = 0; i < N; i ++) {
		parent.push_back(-1);
		dist.push_back(INT_MAX);
		fixed.push_back(false);
	}

	// Make distance value of source vertex as 0 so it is extracted first 
	dist[source] = 0; 
	H.push(distNode{source, dist[0]});

	initMWE(graph); //initialize minimum weight edges of given graph;
//    printf("adj list size is %d\n", graph.adjList.size());
	// Loop for |V| - 1 iterations (our priority queue doesn't support decreaseKey so there will be duplicates in Heap H)

	while (!H.empty())
    {
//	    printf("loop here\n");
//

//	for (int i = 0; i < graph.adjList.size(); i++) {
//        printf("i = %i\n", i);
		// Extract the vertex with minimum dist value
        distNode d = H.top();
		int j = d.node; //pop the minimum distance vertex
        printf("Node is %i\n", j);
		if (!fixed[j]) {
            R.insert(j);
            fixed[j] = true;
            if (parent[j] != -1) {
//                printf("Parent is not null %i\n", parent[j]);
                T.insert(fromTo{j, parent[j]});
            }

            int z;
            while (!R.empty()) {
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
//            printf("processing of edges is done\n");
            while (!Q.empty()) {
                printf("Q is not empty\n");
                std::set<int>::iterator it; //set iterator
                for (it = Q.begin(); it != Q.end(); ++it) {
//				    printf("it is %i Q is %i\n",*it, *Q.end());
//                    int z =0;
                    z = *it;
//                    printf("it is %i Q is %i\n",z, *Q.end());

                    Q.erase(it);
                    printf("Q erased it\n");
                    if (!fixed[z]) {
                        printf("Z is not fixed\n");
                        H.push(distNode{z, dist[z]});
                        printf("Z is %i\n", z);
                        printf("Z dist is %i\n", dist[z]);
                    }
                }
            }
            printf("Q is empty\n");
        }
        H.pop();
	}
//    printf("T size is %i\n",T.size() );
	if (T.size() == graph.adjList.size() -1) {
		return T;
	} else 
		return std::set<fromTo>{}; // return empty tree

}

const char *get_filename_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if (!dot || dot == filename) return "";
    return dot + 1;
}

// Driver program to test above functions 
int main() 
{ 
	// vector of graph edges
	std::vector<edge> edges;
//	edges.push_back(edge{4,5,4});
//	edges.push_back(edge{4,11,8});
//	edges.push_back(edge{5,6,8});
//	edges.push_back(edge{5,11,11});
//	edges.push_back(edge{6,7,7});
//	edges.push_back(edge{6,12,2});
//	edges.push_back(edge{6,9,4});
//	edges.push_back(edge{7,8,9});
//	edges.push_back(edge{7,9,14});
//	edges.push_back(edge{8,9,10});
//	edges.push_back(edge{9,10,2});
//	edges.push_back(edge{10,11,1});
//	edges.push_back(edge{10,12,6});
//	edges.push_back(edge{11,12,7});
    struct dirent *de;  // Pointer for directory entry

    // opendir() returns a pointer of DIR type.
    DIR *dr = opendir(".");

    if (dr == NULL)  // opendir returns NULL if couldn't open directory
    {
        printf("Could not open current directory\n");
        return 0;
    }
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
            int node_count=0;
            int edge_count=0;
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
                       subState=0;
                       break;
                    }
                    if (strncmp(token, "Edges", 2) == 0) {
//                        printf("Section is %s\n", token);
                        state = 2;
                        subState=0;
                        break;
                    }
                    if (strncmp(token, "Operator", 2) == 0) {
//                        printf("Section is %s\n", token);
                        state = 3;
                        subState=0;
                        break;
                    }
                    if (state == 1 && subState==0 ) {
//                        printf("Node is %s\n", token);
                        node_count=node_count+ 1;
                        break;
                    }

                    if (state == 2 && subState == 0) {
                        //                            Source
//                        printf("Source is %s\n", token);
                        source=atoi( token);
                        subState = 1;
                    }else if (state == 2 && subState == 1) {
                        //                            Target
//                        printf("Target is %s\n", token);
                        target =atoi( token);
                        subState = 2;
                    }else if (state == 2 && subState == 2) {
                        //                            Value
//                        printf("Value is %s\n", token);
                        value=(int)atoi( token);
                        edges.push_back(edge{source,target,value});
                        subState = 0;
                        edge_count = edge_count+1;
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
            printf("done with file\n");
            // Maxmum label value of vertices in the given graph, assume 1000
            int N =node_count;
            printf("%i count of nodes\n", node_count);
            // construct graph
            Graph graph(edges, N);
//            printf("created graph\n");

            // print adjacency list representation of graph
//            printGraph(graph);
            printf("print graph\n");

//			//Source vertex as first non empty vertex in adjacency List
//			//Or modify this to take from input file
            int source;
            for(int i = 0; i<nonEmptyIndices.size(); i++) {
                if (nonEmptyIndices[i]) {
                    source = i;
                    break;
                }
            }

            printf("source:%d\n", source);

            printf("Before Prim\n");
//            flush( stdout );
            primMST(graph, N, source);
            printf("After Prim\n");
//            fflush( stdout );

            std::set<fromTo>::iterator it; //set iterator
//            printf("T size:%d\n", T.size());
            printf("Done\n");
            FILE *mst;
            char output[10], filename2[200],extension[5];

            strcpy(output,  "output_");
            strcpy(filename2, strtok(de->d_name, "."));
            strcpy(extension, ".txt");
            strcat(output, filename2);
            strcat(output, extension);
            mst = fopen(output, "w");
            for (it=T.begin(); it!=T.end(); ++it) {
                fromTo e =  *it;
                fprintf(mst,"%d - %d\n", e.from, e.to);
            }
            return 0;
        }
    }
    closedir(dr);
} 

//Reference: https://www.geeksforgeeks.org/prims-mst-for-adjacency-list-representation-greedy-algo-6/
// https://www.techiedelight.com/graph-implementation-using-stl/

