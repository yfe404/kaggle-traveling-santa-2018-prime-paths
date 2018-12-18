#include "problem.hpp"
#include "io.hpp"

int main(int argc, char const *argv[]) {
    cout << "Loading cities from " << argv[1] << "..." << endl;
    vector<City> cities = read_cities(argv[1]);
    cout << "Loaded " << cities.size() << " cities" << endl;

    return 0;
}
