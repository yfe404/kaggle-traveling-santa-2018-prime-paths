#ifndef IO_HPP
#define IO_HPP

#include <fstream>
#include <string>
#include <vector>

#include "problem.hpp"
#include "primes.hpp"
#include "wtf.hpp"

using namespace std;

vector<City<double>> read_cities(string fp) {
    vector<City<double>> cities;
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

vector<City<double>> read_path(vector<City<double>> &cities, string fp) {
    vector<City<double>> path;
    ifstream file(fp);

    if (file.is_open()) {
        string line;

        // Skip first line (Path)
        getline(file, line);

        while (getline(file, line)) {
            // NOTE: This push a copy (important, so that we don't alter cities afterwards)
            path.push_back(cities[stoi(line)]);
        }
    }

    return path;
}

#endif