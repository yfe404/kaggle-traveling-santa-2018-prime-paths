#include <iostream>
#include <vector>
#include <list>
#include <fstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <cassert>
#include <sstream>
#include <sys/time.h>
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <sys/types.h>
#include <random>
#include <numeric>

#include "genome.h"

using namespace std;

template<typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n)
{
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
    return vec;
}


template<class BidiIter >
BidiIter random_unique(BidiIter begin, BidiIter end, size_t num_random) {
    size_t left = std::distance(begin, end);
    while (num_random--) {
        BidiIter r = begin;
        std::advance(r, rand()%left);
        std::swap(*begin, *r);
        ++begin;
        --left;
    }
    return begin;
}

vector<string> list_dir(const char *path) {
  vector<string> files;
  
  struct dirent *entry;
  DIR *dir = opendir(path);
   
  if (dir == NULL) {
    return files;
  }
  while ((entry = readdir(dir)) != NULL) {
    string s_name = string(entry->d_name);
    if (s_name.compare(".") != 0 and s_name.compare("..") != 0)
      files.push_back(s_name);
  }
  closedir(dir);

  return files;
}




void twoOptSwap(
    const vector<int>& inGenes,
    vector<int>& outGenes,
    int iGene1,
    int iGene2);
float fitness(const vector<int>& path, const vector<pair<double,double > > &coords, const vector<int>& primes, int start, int end) ;
/*

while len(ids) > 0:
    edge = ids[0]
    distances = []
    for i in range(len(ppp)):
        distances.append(distance(ppp[i], edge))
    ppp.append(edge)
    insert_pos = np.argmin(distances)+1
   
    if insert_pos < len(ppp)-1:
        ppp = swap_2opt(ppp, insert_pos, len(ppp)-1)
    ids=np.delete(ids, 0, axis=0)

*/

vector<int> read_primes();

vector<int> construct(list<int>& ids, const vector<pair<double,double > > &coords) {

  auto primes = read_primes();
  
  vector<int> path;
  path.push_back(0);

  path.push_back(ids.front());
  ids.pop_front();
  path.push_back(ids.front());
  ids.pop_front();
  
  while(ids.size() > 0) {

    int edge = ids.front();
    vector<double> distances;
    for(int i = 0; i < path.size(); ++i) {
      
      double edgeDistance = sqrt(pow((coords[path[i]].first - coords[edge].first), 2) + 
			  pow((coords[path[i]].second - coords[edge].second), 2));
      distances.push_back(edgeDistance);
    }

    path.push_back(edge);
    int insert_pos = min_element(distances.begin(), distances.end()) - distances.begin();
    int end_pos = path.size()-1;

    if (insert_pos < end_pos) {
      vector<int> new_path(path);
      twoOptSwap(path, new_path, insert_pos, end_pos);
      path.swap(new_path);
    }
    ids.pop_front();
    cout << ids.size() << endl;
  }
  path.push_back(0);

  return path;
}
	     

int isPrime(int n){
    int i;

    if (n < 2)
      return 0;
    
    if (n==2)
        return 1;

    if (n%2==0)
        return 0;

    for (i=3;i<=sqrt(n);i+=2)
        if (n%i==0)
            return 0;

    return 1;
}

vector<int> read_primes() {
  int n;
  vector<int> primes;

  for (int i = 0; i < 200000; ++i) {
    primes.push_back(isPrime(i));
  }

  return primes;
}

vector<pair<double,double > > read_problem() {
  vector<int> id;
  vector<pair<double,double> > coord;

  int n;
  double x, y;

    ifstream infile("kaggle.tsp");

    while(infile >> n) {
      infile >> x >> y;
      coord.push_back(make_pair(x, y));
    }

    return coord;

}


vector<int> read_path(string filename="path.tsp") {

  string line;
  ifstream myfile (filename);
  vector<int> path;

  if (myfile.is_open())
  {
    getline (myfile,line);
    myfile.close();
  }

  else cout << "Unable to open file"; 


  // Build an istream that holds the input string
  istringstream iss(line);

  // Iterate over the istream, using >> to grab floats
  // and push_back to store them in the vector
  std::copy(std::istream_iterator<float>(iss),
	    std::istream_iterator<float>(),
	    std::back_inserter(path));

  return path;
}

vector<Genome> load_generation(int popSize=20) {
  vector<Genome> generation;
  cout << "Loading " << popSize << " genetic sequences........." << endl;

  auto files = list_dir("./genetic_pool");
  random_unique(files.begin(), files.end(), popSize);

  for(int i=0; i<popSize; ++i) {
    auto genetic_seq = read_path(string("./genetic_pool/") + files[i]);
    generation.push_back(Genome(genetic_seq));
  }

  return generation;
}

    typedef unsigned long long timestamp_t;

    static timestamp_t
    get_timestamp ()
    {
      struct timeval now;
      gettimeofday (&now, NULL);
      return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
    }



void twoOptSwap(
    const vector<int>& inGenes,
    vector<int>& outGenes,
    int iGene1,
    int iGene2)
{
    // Take inGenes[0] to inGenes[iGene1 - 1]
    // and add them in order to outGenes

    for (int iGene = 0; iGene <= iGene1 - 1; iGene++)
    {
        outGenes[iGene] = inGenes[iGene];
    }

    // Take inGenes[iGene1] to inGenes[iGene2] and
    // add them in reverse order to outGenes

    int iter = 0;
    for (int iGene = iGene1; iGene <= iGene2; iGene++)
    {
        outGenes[iGene] = inGenes[iGene2 - iter];
        iter++;
    }

    // Take inGenes[iGene2 + 1] to end of inGenes
    // and add them in order to outGenes

    for (int iGene = iGene2 + 1; iGene < inGenes.size(); iGene++)
    {
        outGenes[iGene] = inGenes[iGene];
    }
}

float fitness(const vector<int>& path, const vector<pair<double,double > > &coords, const vector<int>& primes, int start=0, int end=197769) {
  double distance = 0;
  int size = path.size();

  for (int i = start+1; i < size; ++i) {
    if ((i % 10 == 0) && !primes[path[i-1]]) {
      distance += sqrt(pow((coords[path[i-1]].first - coords[path[i]].first), 2) + 
		      pow((coords[path[i-1]].second - coords[path[i]].second), 2)) * 1.1;
    } else {
      distance += sqrt(pow((coords[path[i-1]].first - coords[path[i]].first), 2) + 
		      pow((coords[path[i-1]].second - coords[path[i]].second), 2));
    }
  }
    
    return distance;
}


template<typename T>
bool is_in(vector<T> const &v, T val) {

    auto it = find (v.begin(), v.end(), val);
  
    if (it != v.end())
        return true;
 
    return false;
}

void evaluatePopulation(vector<Genome>& population, const vector<pair<double,double > > &coords, const vector<int>& primes){
  for (auto iter = population.begin(); iter != population.end(); ++iter) {
    iter->set_fitness(fitness(iter->get_phenotype(), coords, primes ));
  }

  std::sort(population.begin(), population.end());

  cout << "Best individual: " << population[population.size()-1].get_fitness() << endl;
  cout << "Worst individual: " << population[0].get_fitness() << endl; 
}

// @todo: replace with mte
double randMToN(double M, double N)
{
    return M + (rand() / ( RAND_MAX / (N-M) ) ) ;  
}

Genome crossover(const Genome &parent1, const Genome &parent2) {
    
    vector<int> child;
    vector<int> city_included;
    int size = parent1.get_phenotype().size();
    bool STOP = false;
    
    int rnd_city = randMToN(1, size);
    
    child.push_back(rnd_city);
    
    auto it = find(parent1.get_phenotype().begin(), parent1.get_phenotype().end(), rnd_city);
    int rnd_city_in_parent1 = it - parent1.get_phenotype().begin();
    
    it = find(parent2.get_phenotype().begin(), parent2.get_phenotype().end(), rnd_city);
    int rnd_city_in_parent2 = it - parent2.get_phenotype().begin();
    
    city_included.push_back(rnd_city);
    
    int idx1 = rnd_city_in_parent1 + 1;
    int idx2 = rnd_city_in_parent2 - 1;
    
    while(!STOP) {
        if (idx1 < size-1) { // -1 for the 0 
            if(is_in(city_included, parent1.get_phenotype()[idx1])) {
                STOP = true;
                break;
            }
            child.insert(child.begin(), parent1.get_phenotype()[idx1]);
            city_included.push_back(parent1.get_phenotype()[idx1]);
            idx1++;
        } else {break;}

        if(idx2 > 0) { // > strict for the 0 
            if(is_in(city_included, parent2.get_phenotype()[idx2])) {
                STOP = true;
                break;
            }
            child.insert(child.begin(), parent2.get_phenotype()[idx2]);
            city_included.push_back(parent2.get_phenotype()[idx2]);
            idx2--;
        }
    }
    
    for(int i = 1; i < size-1; ++i) { // -1 for the zero
        if (!is_in(city_included, parent2.get_phenotype()[i])) {
            child.push_back(parent2.get_phenotype()[i]);
        }
    }
        
    child.push_back(0);
    child.insert(child.begin(), 0);
    
    return Genome(child);

}


Genome mutate(Genome& genome) {

  int size = genome.get_phenotype().size();
  
  int i = randMToN(1, (int)(0.25*size));
  int ii = i + 1;

  int j = randMToN(ii+1, (int)(0.50*size));
  int jj = j + 1;

  int k = randMToN(jj+1, (int)(0.75*size));
  int kk = k + 1;

  int l = randMToN(kk+1, size-2);
  int ll = l + 1;

  cout << i << " " << ii <<  " " << j << " " << jj << " " <<  k << " " << kk << " " << l << " " << ll << endl;

  vector<int> phenotype;

  cout << "WTF"  << endl;

  std::vector<int> sub_vec1 = slice(genome.get_phenotype(), 0, i);

  phenotype.insert(
      phenotype.end(),
      std::make_move_iterator(sub_vec1.begin()),
      std::make_move_iterator(sub_vec1.end())
    );

  phenotype.push_back(genome.get_phenotype()[kk]);


  std::vector<int> sub_vec2 = slice(genome.get_phenotype(), kk+1, l);

  phenotype.insert(
      phenotype.end(),
      std::make_move_iterator(sub_vec2.begin()),
      std::make_move_iterator(sub_vec2.end())
    );

  phenotype.push_back(genome.get_phenotype()[jj]);

  std::vector<int> sub_vec3 = slice(genome.get_phenotype(), jj+1, k);

  phenotype.insert(
      phenotype.end(),
      std::make_move_iterator(sub_vec3.begin()),
      std::make_move_iterator(sub_vec3.end())
    );

  phenotype.push_back(genome.get_phenotype()[ii]);


    std::vector<int> sub_vec4 = slice(genome.get_phenotype(), ii+1, j);

  phenotype.insert(
      phenotype.end(),
      std::make_move_iterator(sub_vec4.begin()),
      std::make_move_iterator(sub_vec4.end())
    );

  phenotype.push_back(genome.get_phenotype()[ll]);

  std::vector<int> sub_vec5 = slice(genome.get_phenotype(), ll+1, size-1);

  phenotype.insert(
      phenotype.end(),
      std::make_move_iterator(sub_vec5.begin()),
      std::make_move_iterator(sub_vec5.end())
    );


  return Genome(phenotype);

}


template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

vector<unsigned long> getNN(int cityID, const vector<pair<double,double > > &coords, int n=20) {

  vector<double> distances;

  for(int i = 1; i < coords.size(); ++i){
    if (i == cityID){continue;}
    double edgeDistance = sqrt(pow((coords[cityID].first - coords[i].first), 2) + 
		      pow((coords[cityID].second - coords[i].second), 2));
    distances.push_back(edgeDistance);
  }

  auto dd = sort_indexes(distances);
  return slice(dd, 0, n-1);

}


void run2kopt(Genome& genome, const vector<pair<double,double > > &coords, const vector<int>& primes) {
  
  float _fitness = 0.0;
  bool hasImproved = true;
  double swappedGenesFitness = 0.0;
  auto path = vector<int>(genome.get_phenotype());
  int size = path.size();
  
  _fitness = fitness(path, coords, primes);

  vector<int> swappedGenes(size);

  int k = 0;
  while (hasImproved) {
    for (int iGene1 = 1; iGene1 < path.size() - 1; ++iGene1) {
      auto nearests = getNN(iGene1, coords, 30);
      for (int i = 0; i < 30; i++) {
	int iGene2 = nearests[i];
	if ((iGene2 <= iGene1)) continue;
	timestamp_t t0 = get_timestamp();
	double prevCost = fitness(path, coords, primes, iGene1-1, iGene2+1);
	twoOptSwap(path, swappedGenes, iGene1, iGene2);
	double newCost = fitness(swappedGenes, coords, primes, iGene1-1, iGene2+1); 
	
	if (newCost < prevCost) {
	  std::cout << std::fixed << std::setprecision(2);
	  cout << "Cost on portion (old) " << prevCost << endl;
	  cout << "Cost on portion (new) " << newCost << endl;

	   path.swap(swappedGenes);
	   cout << "Old overall fitness " << _fitness << endl ;
	  _fitness = fitness(path, coords, primes);
	    prevCost = newCost;
	  hasImproved = true;
	  cout << "New overall fitness " << _fitness << endl << endl;
	}
	else { hasImproved = false; }

      }

    }
  }

  genome.set_phenotype(path);
}



int main() {

  auto coords = read_problem();
  auto primes = read_primes();

  auto path = read_path();
  list<int> ids;
  copy( path.begin(), path.end()-1, back_inserter( ids ) );


    
  cout << "-._    _.--'\"`'--._    _.--'\"`'--._    _.--'\"`'--._    _" << endl;
  cout << "'-:`.'|`|\"':-.  '-:`.'|`|\"':-.  '-:`.'|`|\"':-.  '.` : '.   " << endl;
  cout << "  '.  '.  | |  | |'.  '.  | |  | |'.  '.  | |  | |'.  '.:   '.  '." << endl;
  cout << "  : '.  '.| |  | |  '.  '.| |  | |  '.  '.| |  | |  '.  '.  : '.  `." << endl;
  cout << "  '   '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.'   `." << endl;
  cout << "         `-..,..-'       `-..,..-'       `-..,..-'       `         `" << endl;

  float MUTRATE = 0.9;
  int NGEN = 1000;
  int POPSIZE = 20;
  int SEED = 42;

  mt19937 mte(SEED);  // mt19937 is a standard mersenne_twister_engine
  
  cout << "Using Mersenne Twister Engine (Matsumoto and Nishimura)." << endl;
  cout << "Setting seed to " << SEED << endl;
  cout << "Mutation rate set to " << MUTRATE << endl;
  cout << "Number of generations set to " << NGEN << endl;
  cout << "Population size set to " << POPSIZE << endl;
  cout << "Using Greedy Subtour Crossover V.2 (GSX-2)" << endl;
  cout << "Using Double-Bridge Mutation" << endl;

  cout << "-._    _.--'\"`'--._    _.--'\"`'--._    _.--'\"`'--._    _" << endl;
  cout << "'-:`.'|`|\"':-.  '-:`.'|`|\"':-.  '-:`.'|`|\"':-.  '.` : '.   " << endl;
  cout << "  '.  '.  | |  | |'.  '.  | |  | |'.  '.  | |  | |'.  '.:   '.  '." << endl;
  cout << "  : '.  '.| |  | |  '.  '.| |  | |  '.  '.| |  | |  '.  '.  : '.  `." << endl;
  cout << "  '   '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.'   `." << endl;
  cout << "         `-..,..-'       `-..,..-'       `-..,..-'       `         `" << endl;

  cout << "Initializing population....." << endl ;
  auto population = load_generation();

  evaluatePopulation(population, coords, primes);

  mutate(population[0]);
  crossover(population[0], population[1]);

  evaluatePopulation(population, coords, primes);
  
  
  for (int i = 0; i < NGEN; ++i) {
    Genome child;
    
    cout << "========== GENERATTION " << i << " ==========" << endl;
    double rnd = (double)mte() / (double)mte.max();
    if (rnd < MUTRATE) {
      cout << "Genetic pool initialized for mutation..." << endl;
      cout << "Selecting parent sequence..." << endl;
      cout << "Starting mutation..." << endl;
      child = mutate(population[POPSIZE-1]);
    } else {
      cout << "Genetic pool initialized for crossover..." << endl;
      cout << "Selecting parent sequences for crossover..." << endl;
      cout << "Starting Crossover..." << endl;
      child = crossover(population[POPSIZE-1], population[POPSIZE-2]);
    }

    cout << "Improvement Heuristic started..." << endl;

    run2kopt(child, coords, primes);
    
    cout << "Checking improvement of offspring genetic sequence obtained..." << endl;
    if ( population[0] < child) {
      population[0] = child;
    }

    evaluatePopulation(population, coords, primes);
    
  }

}
