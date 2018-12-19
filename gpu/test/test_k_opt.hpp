#include <cxxtest/TestSuite.h>
#include <algorithm>

#include "../src/problem.hpp"
#include "../src/k_opt.hpp"
#include "../src/io.hpp"

class TestKopt : public CxxTest::TestSuite {
    public:
    void testTwoOptScore(void) {
        auto cities = read_cities<double>("../test/cities.csv");
        auto path = read_path(cities, "../test/1516773.csv");
        // Edge cases are %10 +/- indices
        for (int i = 8; i <= 12; i++) {
            for (int j = 18; j <= 22; j++) {
                vector<City<double>> path2(path);
                // C++ reverse is [first, last)
                // Julia reverse is [first, last]
                // Hence the +1
                reverse(path2.begin()+i, path2.begin()+j+1);
                // Allow errors < 1e-6 since two_opt_score may be more precise than
                // the score difference due to numerical errors
                TS_ASSERT_DELTA(
                    two_opt_score(&path[0], i, j),
                    score(path2) - score(path),
                    1e-6
                );
            }
        }
    }
};
