#ifndef GENOME_H
#define GENOME_H

#include <vector>

using std::vector;


/**
  @class Genome class represents a solution to TSP
  @brief Holds a tour represented as a list of cities and the distance (also refered as score or fitness).
**/


class Genome {
 public:
  Genome(vector<int> phenotype);
  vector<int> get_phenotype();

 private:
  vector<int> phenotype; 
  int fitness;
};

#endif // GENOME_H
