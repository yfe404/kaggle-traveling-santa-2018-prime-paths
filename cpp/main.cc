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


 #include <sys/time.h>
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


float fitness(const vector<int>& path, const vector<pair<double,double > > &coords) {

  //  int currentIndex = 0;
  double distance = 0;
  int size = path.size();

  float discount = 0.0;
  float edgeDistance = 0.0;

  //  timestamp_t t0 = get_timestamp();
  for (int i = 1; i < size; ++i) {
    
    if ((((i)%10) == 0) && (!isPrime(path[i-1]))) discount = 1.1;

    edgeDistance = sqrt(pow((coords[path[i-1]].first - coords[path[i]].first), 2) + 
		      pow((coords[path[i-1]].second - coords[path[i]].second), 2));

    edgeDistance *= discount;
    distance += edgeDistance;
    discount = 1.0;
  }

  // timestamp_t t1 = get_timestamp();

  //double secs = (t1 - t0) / 1000000.0L;
    
    return distance;
}


int main() {

  auto path = read_path();
  auto coords = read_problem();

  /*======================================================================================================================================================================================================== LOCAL SEARCH ======================================================================================================================================================================================================== */

  float _fitness = fitness(path, coords);
  bool hasImproved = true;
  double swappedGenesFitness = 0.0;


  cout << "Fitness score of initial genetic sequence: " << _fitness << endl;  
  
  vector<int> swappedGenes(path.size());

  while (hasImproved) {
    for (int iGene1 = 1; iGene1 < path.size() - 1; iGene1++)
      for (int iGene2 = iGene1 + 1; iGene2 < path.size(); iGene2++) {
	twoOptSwap(path, swappedGenes, iGene1, iGene2);
	swappedGenesFitness = fitness(swappedGenes, coords);

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
