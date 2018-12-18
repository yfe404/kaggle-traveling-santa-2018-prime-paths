#ifndef PRIMES_HPP
#define PRIMES_HPP

#include <cmath>

bool is_prime(int n) {
    int i;
    if (n < 2)
        return false;
    if (n == 2)
        return true;
    if (n % 2 == 0)
        return false;
    for (i = 3; i <= sqrt((float)n); i += 2)
        if (n % i == 0)
            return false;
    return true;
}

#endif