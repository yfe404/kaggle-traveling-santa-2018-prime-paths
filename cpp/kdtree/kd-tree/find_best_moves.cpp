#include <iostream>
#include <string>
#include <vector>
#include <algorithm> 

using namespace std;

struct delta_t {
    delta_t(int start, int end, double profit): start(start), end(end), profit(profit) {}
    int start; // start position
    int end; // end postion 
    double profit; // value of the move, the higher the better
    
    bool operator < (const delta_t& delta) const
    {
        return (end < delta.end);
    }
    
} ;


// Find the latest interval (in sorted array) that doesn't 
// conflict with the moves[i]. If there is no compatible interval, 
// then it returns -1. 
int latestNonConflict(const std::vector<delta_t>& moves, int i) 
{ 
    for (int j=i-1; j>=0; j--) 
    { 
        if (moves[j].end < moves[i-1].start - 1)
            return j; 
    } 
    return -1; 
} 
  
// A recursive function that returns the maximum possible 
// interval from given vector of delta_t objects. The vector of delat_t must 
// be sorted according to finish index. 
double findMaximumProfit(const std::vector<delta_t>& moves, int n) {  
  // Create an array to store solutions of subproblems.  table[i] 
  // stores the profit for moves till moves[i] (including moves[i]) 
  double *table = new double[n]; 
  table[0] = moves[0].profit;
  
  // Fill entries in M[] using recursive property 
  for (int i=1; i<n; i++) { 
    // Find profit including the current job 
    double inclProf = moves[i].profit; 
    int l = latestNonConflict(moves, i); 
    if (l != -1) 
      inclProf += table[l]; 
      
    // Store maximum of including and excluding 
    table[i] = max(inclProf, table[i-1]); 
  } 
  
  // Store result and free dynamic memory allocated for table[] 
  double result = table[n-1]; 
  delete[] table; 
  
  return result;  
} 
 

int main()
{   
    delta_t d0(1,2,50);
    delta_t d1(6,19,100);
    delta_t d2(3,5,20);
    delta_t d3(4,100,200);
    
    std::vector<delta_t> moves;
    
    moves.push_back(d0);
    moves.push_back(d1);
    moves.push_back(d2);
    moves.push_back(d3);

    // Sort jobs according to finish time
    std::sort (moves.begin(), moves.end());
      
    for (unsigned int i = 0; i < moves.size(); ++i) {
        cout << moves[i].end << endl;    
    }
    
    cout << "The maximum profit is " << findMaximumProfit(moves, moves.size()) << "." << endl;
 
}
