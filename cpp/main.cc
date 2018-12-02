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

#include "genome.h"

using namespace std;



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

  //  int currentIndex = 0;
  double distance = 0;
  int size = path.size();

  //  cout << size << " " << start << " " <<  end <<  endl;
  
  for (int i = start+1; i < size; ++i) {
    float edgeDistance = 0.0;
    //    cout << i << endl;
    edgeDistance = sqrt(pow((coords[path[i-1]].first - coords[path[i]].first), 2) + 
		      pow((coords[path[i-1]].second - coords[path[i]].second), 2));
    //    cout << "BT1" << endl;

    //    cout << i+step-1 << endl;
    if ((((i)%10) == 0) && (!primes[path[i-1]])) {
      edgeDistance *= 1.1;
    }

    //    cout << "BT2" << endl;
    distance += edgeDistance;
  }
    
    return distance;
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

void mutate(Genome& genome) {

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

  vector<int> phenotype(size);

  cout << "WTF"  << endl;


  std::copy(begin(genome.get_phenotype()), begin(genome.get_phenotype())+i, phenotype.begin());

    cout << "WTF"  << endl;
  phenotype.push_back(genome.get_phenotype()[kk]);
  
  std::copy(begin(genome.get_phenotype())+kk+1, begin(genome.get_phenotype())+l,
            end(phenotype));

  phenotype.push_back(genome.get_phenotype()[jj]);

  std::copy(begin(genome.get_phenotype())+jj+1, begin(genome.get_phenotype())+k,
            end(phenotype));

  phenotype.push_back(genome.get_phenotype()[ii]);

  std::copy(begin(genome.get_phenotype())+ii+1, begin(genome.get_phenotype())+j,
            end(phenotype));
  
  phenotype.push_back(genome.get_phenotype()[ll]);

  std::copy(begin(genome.get_phenotype())+ll+1, end(genome.get_phenotype()),
            end(phenotype));

  /*
  for (int i = 0; i < size; ++i) {
      cout << phenotype[i] << endl;
  }
  */
  genome.set_phenotype(phenotype);

 }



int main() {

  auto population = load_generation();
  auto coords = read_problem();
  auto primes = read_primes();


  
  auto path = read_path();
  list<int> ids;
  copy( path.begin(), path.end()-1, back_inserter( ids ) );


  cout << "Initializing population....." << endl ;
  evaluatePopulation(population, coords, primes);

  mutate(population[0]);

  evaluatePopulation(population, coords, primes);
    
  cout << "-._    _.--'\"`'--._    _.--'\"`'--._    _.--'\"`'--._    _" << endl;
  cout << "'-:`.'|`|\"':-.  '-:`.'|`|\"':-.  '-:`.'|`|\"':-.  '.` : '.   " << endl;
  cout << "  '.  '.  | |  | |'.  '.  | |  | |'.  '.  | |  | |'.  '.:   '.  '." << endl;
  cout << "  : '.  '.| |  | |  '.  '.| |  | |  '.  '.| |  | |  '.  '.  : '.  `." << endl;
  cout << "  '   '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.'   `." << endl;
  cout << "         `-..,..-'       `-..,..-'       `-..,..-'       `         `" << endl;

  float MUTRATE = 0.5;
  int NGEN = 5;
  int POPSIZE = 100;
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

  
  for (int i = 0; i < NGEN; ++i) {
    cout << "========== GENERATTION " << i << " ==========" << endl;
    double rnd = (double)mte() / (double)mte.max();
    if (rnd < MUTRATE) {
      cout << "Genetic pool initialized for mutation..." << endl;
      cout << "Selecting parent sequence..." << endl;
      cout << "Starting mutation..." << endl;
    } else {
      cout << "Genetic pool initialized for crossover..." << endl;
      cout << "Selecting parent sequences for crossover..." << endl;
      cout << "Starting Crossover..." << endl;
    }

    cout << "Improvement Heuristic started..." << endl;
    cout << "Checking improvement of offspring genetic sequence obtained..." << endl;
    
  }

  /*

Begin
 For each subpopulation do
   Initialize subpopulation by tour
   construction heuristic;
 Repeat {
   For each subpopulation do {
If (rand() < mutation rate) {
Select one parent p using linear ranking; Mutation(p, c);
    } Else {
      Select two parents p1, p2 using linear
      ranking;
      Crossover(p1, p2, c);
}
Tour improvement heuristic(c);
Replace c to the worst parent if fitter;
}
At predefined migration interval do
    Migration between subpopulations
 }
 Until converged;
End
   */
  /*
  for (auto iter = generation.begin(); iter != generation.end(); ++iter) {
    auto genetic_seq = *iter;
    //    std::cout << fitness (genetic_seq, coords, primes)<< endl;
  }
  */

  /*
  path = construct(ids, coords);
  float _ffitness = fitness(path, coords, primes);
  std::cout << std::fixed << std::setprecision(2);
  cout << "Fitness score of initial genetic sequence: " << _ffitness << endl;
  */
  exit(0);

  
  /*======================================================================================================================================================================================================== LOCAL SEARCH ======================================================================================================================================================================================================== */

  float _fitness = 0.0;
  bool hasImproved = true;
  double swappedGenesFitness = 0.0;

  timestamp_t t0 = get_timestamp();
  _fitness = fitness(path, coords, primes);
  timestamp_t t1 = get_timestamp();

  double secs = (t1 - t0) / 1000000.0L;

  std::cout << std::fixed << std::setprecision(2);
  cout << "Fitness score of initial genetic sequence: " << _fitness << endl;
  cout << "found in " << secs << " seconds" << endl;
  
  vector<int> swappedGenes(path.size());

  int k = 0;
  while (hasImproved) {
    for (int iGene1 = 1; iGene1 < path.size() - 1; iGene1++) {
      //      cout << iGene1 << endl;
      for (int iGene2 = iGene1 + 1; iGene2 < path.size(); iGene2++) {
	//	if (((iGene1-1) %10) != 0) continue;
	if ((iGene2 > iGene1 + 20)) continue;
	//	cout << iGene2 << endl;
	//	if ((iGene1 != 21)) continue;
	//	cout << iGene1 << " " << iGene2 << endl;
	timestamp_t t0 = get_timestamp();
	double prevCost = fitness(path, coords, primes, iGene1-1, iGene2+1);
	//	cout << "BP" << endl;
	twoOptSwap(path, swappedGenes, iGene1, iGene2);
	//	cout << "BP" << endl;
	//	cout << iGene1 << " " << iGene2 << endl;
	double newCost = fitness(swappedGenes, coords, primes, iGene1-1, iGene2+1); // segfault 
	//	cout << "BP" << endl;


	
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
  
  return 0;
}
