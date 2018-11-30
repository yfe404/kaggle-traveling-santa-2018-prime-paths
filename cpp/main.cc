#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <cassert>
#include <sstream>

#include <cmath>


using namespace std;


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

vector<int> read_path() {

  string line;
  ifstream myfile ("path.tsp");
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




 #include <sys/time.h>
    typedef unsigned long long timestamp_t;

    static timestamp_t
    get_timestamp ()
    {
      struct timeval now;
      gettimeofday (&now, NULL);
      return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
    }


double distanceEdges(const vector<int>& path, const vector<pair<double,double > > &coords, int start, int end) {

  float discount = 1.0 + 0.1 * ((!isPrime(path[start])) && (path[end] % 10 == 0));
  float edgeDistance = 0.0;

  edgeDistance = sqrt(pow((coords[path[start]].first - coords[path[end]].first), 2) + 
		      pow((coords[path[start]].second - coords[path[end]].second), 2));

  edgeDistance *= discount;

  return edgeDistance;
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


float fitness(const vector<int>& path, const vector<pair<double,double > > &coords, const vector<int>& primes, int step=0) {

  //  int currentIndex = 0;
  double distance = 0;
  int size = path.size();

  for (int i = 1; i < size; ++i) {
    float edgeDistance = 0.0;
    
    edgeDistance = sqrt(pow((coords[path[i-1]].first - coords[path[i]].first), 2) + 
		      pow((coords[path[i-1]].second - coords[path[i]].second), 2));

    if ((((i+step)%10) == 0) && (!primes[path[(i+step)-1]])) {
      edgeDistance *= 1.1;
    }

    distance += edgeDistance;
  }

    
    return distance;
}


int main() {

  auto path = read_path();
  auto coords = read_problem();
  auto primes = read_primes();

  /*======================================================================================================================================================================================================== LOCAL SEARCH ======================================================================================================================================================================================================== */

  float _fitness = 0.0;
  bool hasImproved = true;
  double swappedGenesFitness = 0.0;

  timestamp_t t0 = get_timestamp();
  _fitness = fitness(path, coords, primes);
  timestamp_t t1 = get_timestamp();

  double secs = (t1 - t0) / 1000000.0L;


  cout << "Fitness score of initial genetic sequence: " << _fitness << endl;
  cout << "found in " << secs << " seconds" << endl;
  
  vector<int> swappedGenes(path.size());

  double prevEdge1Cost = 0.0;
  double prevEdge2Cost = 0.0;

  
  while (hasImproved) {
    for (int iGene1 = 1; iGene1 < path.size() - 1; iGene1++)
      for (int iGene2 = iGene1 + 1; iGene2 < path.size(); iGene2++) {
	twoOptSwap(path, swappedGenes, iGene1, iGene2);
	//	swappedGenesFitness = fitness(swappedGenes, coords);

	prevEdge1Cost = distanceEdges(path, coords, iGene1, iGene2);
	prevEdge2Cost = distanceEdges(swappedGenes, coords, iGene1, iGene2);
	
	cout << prevEdge1Cost << endl;
	cout << prevEdge2Cost << endl;

	exit(0);

	
	if (swappedGenesFitness < _fitness) {
	  path.swap(swappedGenes);
	  _fitness = swappedGenesFitness;
	  hasImproved = true;
	  cout << _fitness << endl;
	}
	else { hasImproved = false; }
      }
  }
  
  return 0;
}
