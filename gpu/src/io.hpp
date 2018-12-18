#ifndef IO_HPP
#define IO_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "problem.hpp"
#include "primes.hpp"
#include "wtf.hpp"

using namespace std;

template <typename T>
vector<City<T>> read_cities(string fp) {
    vector<City<T>> cities;
    ifstream file(fp);

    if (file.is_open()) {
        string line;
        
        // Skip first line (CityId,X,Y)
        getline(file, line);

        while (getline(file, line)) {
            vector<string> tokens = split(line, ',');
            int id = stoi(tokens[0]);
            T x = stod(tokens[1]);
            T y = stod(tokens[2]);
            cities.push_back({id, is_prime(id), {x, y}});
        }
        file.close();
    } else {
        cerr << "Failed to open " << fp << endl;
    }

    return cities;
}

template <typename T>
vector<City<T>> read_path(vector<City<T>> &cities, string fp) {
    vector<City<T>> path;
    ifstream file(fp);

    if (file.is_open()) {
        string line;

        // Skip first line (Path)
        getline(file, line);

        while (getline(file, line)) {
            // NOTE: This push a copy (important, so that we don't alter cities afterwards)
            path.push_back(cities[stoi(line)]);
        }
    } else {
        cerr << "Failed to open " << fp << endl;
    }

    return path;
}

// TODO: Constrain to City iterators (possible in C++ ?)
template <class InputIt>
void write_path(InputIt first, InputIt last, string fp) {
    ofstream file(fp);
    stringstream ss;

    ss << "Path" << endl;
    for(; first != last; first++) {
        ss << first->i << endl;
    }

    if (file.is_open()) {
        file << ss.str();
        file.close();
    } else {
        cerr << "Failed to open " << fp << endl;
    }
}

#endif