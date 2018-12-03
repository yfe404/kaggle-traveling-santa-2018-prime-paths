#include <iostream> 
using namespace std; 
  
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
  std::cout << "Processing genetic sequences....." << endl;
  return 0;

} 
