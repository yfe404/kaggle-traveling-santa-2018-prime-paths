#ifndef IO_HPP
#define IO_HPP

#include <fstream>
#include <string>
#include <vector>

#include "problem.hpp"
#include "primes.hpp"
#include "wtf.hpp"

using namespace std;

vector<City> read_cities(string fp) {
    vector<City> cities;
    ifstream file(fp);

    if (file.is_open()) {
        string line;
        
        // Skip first line (CityId,X,Y)
        getline(file, line);

        while (getline(file, line)) {
            vector<string> tokens = split(line, ',');
            int id = stoi(tokens[0]);
            double x = stod(tokens[1]);
            double y = stod(tokens[2]);
            cities.push_back({id, is_prime(id), {x, y}});
        }
        file.close();
    }
    return cities;
}

#endif