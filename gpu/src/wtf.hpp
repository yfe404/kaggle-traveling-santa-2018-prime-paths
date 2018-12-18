#ifndef WTF_HPP
#define WTF_HPP

#include <string>
#include <vector>
#include <sstream>
#include <iostream>

// https://thispointer.com/how-to-split-a-string-in-c/
// WTF why is it so complicated to split a string in C++ ???
std::vector<std::string> split(std::string strToSplit, char delimeter)
{
    std::stringstream ss(strToSplit);
    std::string item;
    std::vector<std::string> splittedStrings;
    while (std::getline(ss, item, delimeter))
    {
       splittedStrings.push_back(item);
    }
    return splittedStrings;
}

#endif