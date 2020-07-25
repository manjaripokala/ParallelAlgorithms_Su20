#include <vector>

// Returns the euclidian lenght of 'vector'.
double euclidean_length(std::vector<double> vector);

// Returns a sorted vector that contains all the unique elements in 'sorted_vector'.
//
// NOTE: non-const vectors are not thread-safe.
std::vector<int64_t> discard_duplicates(std::vector<int64_t> sorted_vector);
