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
    if (n == 1) return moves[0].profit;
     
     // Find profit when current move is inclueded 
    int inclProf = moves[n-1].profit; 
    int i = latestNonConflict(moves, n); 
    if (i != -1) inclProf += findMaximumProfit(moves, i+1); 
  
    // Find profit when current job is excluded 
    int exclProf = findMaximumProfit(moves, n-1); 
  
    return max(inclProf,  exclProf); 
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
    
    // STEP 1 - Sorting intervals according to end index
    std::sort (moves.begin(), moves.end());
    
    for (unsigned int i = 0; i < moves.size(); ++i) {
        cout << moves[i].end << endl;    
    }
    
    cout << "The maximum profit is " << findMaximumProfit(moves, moves.size()) << "." << endl;
 
}
