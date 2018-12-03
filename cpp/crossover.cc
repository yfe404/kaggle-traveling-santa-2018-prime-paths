#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <cassert>
#include <sstream>
#include <sys/time.h>
#include <cmath>
#include <cassert>

using namespace std;

typedef vector<int> Genome;

vector<int> read_path(string filename);
double randMToN(double M, double N);
template<typename T>
bool is_in(vector<T> const &v, T val);
Genome crossover(const Genome &parent1, const Genome &parent2);
void print_seq();
  
int main(int argc, char** argv) 
{
  // Check the number of parameters
  if (argc < 3) {
    // Tell the user how to run the program
    std::cerr << "Usage: " << argv[0] << " PATH_TO_PARENT1.TSP PATH_TO_PARENT2.TSP" << std::endl;
    /* "Usage messages" are a conventional way of telling the user
     * how to run a program if they enter the command incorrectly.
     */
    return 1;
  }

  print_seq();
  std::cout << "Processing genetic sequences....." << endl;
  std::cout << "Reading first sequence...." << endl;
  auto genetic_seq1 = read_path(argv[1]);
  std::cout << "Reading second sequence...." << endl;
  auto genetic_seq2 = read_path(argv[2]);
  std::cout << "Crossover started...." << endl;
  auto offspring = crossover(genetic_seq1, genetic_seq2);
  std::cout << endl;
  std::cout << "[DONE] GENETIC SEQUENCE PRODUCED!" << endl;
  
  return 0;

} 


vector<int> read_path(string filename) {

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



Genome crossover(const Genome &parent1, const Genome &parent2) {
    
    Genome child;
    Genome city_included;
    int size1 = parent1.size();
    int size2 = parent2.size();
    int size = size1;
    bool STOP = false;
    int rnd_city = randMToN(1, size);

    assert(size1==size2);
    assert(parent1[0] == parent1[size1-1]);
    assert(parent2[0] == parent2[size2-1]);
    
    child.push_back(rnd_city);
    
    auto it = find(parent1.begin(), parent1.end(), rnd_city);
    int rnd_city_in_parent1 = it - parent1.begin();
    
    it = find(parent2.begin(), parent2.end(), rnd_city);
    int rnd_city_in_parent2 = it - parent2.begin();
    
    city_included.push_back(rnd_city);
    
    int idx1 = rnd_city_in_parent1 + 1;
    int idx2 = rnd_city_in_parent2 - 1;
    
    while(!STOP) {
        if (idx1 < size-1) { // -1 for the 0 
            if(is_in(city_included, parent1[idx1])) {
                STOP = true;
                break;
            }
            child.insert(child.begin(), parent1[idx1]);
            city_included.push_back(parent1[idx1]);
            idx1++;
        } else {break;}

        if(idx2 > 0) { // > strict for the 0 
            if(is_in(city_included, parent2[idx2])) {
                STOP = true;
                break;
            }
            child.insert(child.begin(), parent2[idx2]);
            city_included.push_back(parent2[idx2]);
            idx2--;
        }
    }
    
    for(int i = 1; i < size-1; ++i) { // -1 for the zero
        if (!is_in(city_included, parent2[i])) {
            child.push_back(parent2[i]);
        }
    }
        
    child.push_back(0);
    child.insert(child.begin(), 0);

    int size_child = child.size();
    assert(size_child == size1);
    assert(child[0] == child[size1-1]);
    
    return child;
}


double randMToN(double M, double N)
{
    return M + (rand() / ( RAND_MAX / (N-M) ) ) ;  
}


template<typename T>
bool is_in(vector<T> const &v, T val) {

    auto it = find (v.begin(), v.end(), val);
  
    if (it != v.end())
        return true;
 
    return false;
}


void print_seq() {
    cout << "-._    _.--'\"`'--._    _.--'\"`'--._    _.--'\"`'--._    _" << endl;
  cout << "'-:`.'|`|\"':-.  '-:`.'|`|\"':-.  '-:`.'|`|\"':-.  '.` : '.   " << endl;
  cout << "  '.  '.  | |  | |'.  '.  | |  | |'.  '.  | |  | |'.  '.:   '.  '." << endl;
  cout << "  : '.  '.| |  | |  '.  '.| |  | |  '.  '.| |  | |  '.  '.  : '.  `." << endl;
  cout << "  '   '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.'   `." << endl;
  cout << "         `-..,..-'       `-..,..-'       `-..,..-'       `         `" << endl;

}
