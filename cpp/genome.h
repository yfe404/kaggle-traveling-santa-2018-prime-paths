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
  Genome() {};
  Genome(vector<int> phenotype);

  const vector<int>&  get_phenotype() const {return phenotype;};
  void set_phenotype(vector<int> phenotype) {this->phenotype = phenotype;}; 

  bool operator < (const Genome& otherGenome) const
  {
    return (this->fitness > otherGenome.fitness);
  }

  long get_fitness() { return this->fitness; };
  void set_fitness(long fitness) { this->fitness = fitness;}; 
  
 private:
  vector<int> phenotype; 
  long fitness = 999999999999;
};

#endif // GENOME_H
