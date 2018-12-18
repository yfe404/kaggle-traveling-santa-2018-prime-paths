#include "problem.hpp"
#include "io.hpp"

int main(int argc, char const *argv[]) {
    cout << "Loading cities from " << argv[1] << "..." << endl;
    vector<City<double>> cities = read_cities(argv[1]);
    cout << "Loaded " << cities.size() << " cities" << endl;

    vector<City<double>> path = read_path(cities, argv[2]);
    cout << &cities[0] << endl;
    cout << &path[0] << endl;

    return 0;
}
