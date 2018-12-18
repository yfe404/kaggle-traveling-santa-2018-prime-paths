#ifndef PRIMES_HPP
#define PRIMES_HPP

#include <cmath>

int is_prime(int n) {
    int i;
    if (n < 2)
        return 0;
    if (n == 2)
        return 1;
    if (n % 2 == 0)
        return 0;
    for (i = 3; i <= sqrt((float)n); i += 2)
        if (n % i == 0)
            return 0;
    return 1;
}

#endif