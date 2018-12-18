#include <cxxtest/TestSuite.h>
#include "../src/io.hpp"

class TestIO : public CxxTest::TestSuite {
    public:
    void testReadCities(void) {
        vector<City<double>> cities = read_cities("../test/cities.csv");
        TS_ASSERT_EQUALS(cities.size(), 197769);
        TS_ASSERT_EQUALS(cities[2].i, 2);
        TS_ASSERT_EQUALS(cities[2].p, true);
        TS_ASSERT_EQUALS(cities[2].xy.x, 3454.15819771172);
        TS_ASSERT_EQUALS(cities[2].xy.y, 2820.05301124811);
        TS_ASSERT_EQUALS(cities[197768].i, 197768);
        TS_ASSERT_EQUALS(cities[197768].p, false);
    }

    void testReadPath(void) {
        vector<City<double>> cities = read_cities("../test/cities.csv");
        vector<City<double>> path = read_path(cities, "../test/1516773.csv");
        TS_ASSERT_EQUALS(path.size(), 197770);
        TS_ASSERT_EQUALS(path[0].i, 0);
        TS_ASSERT_EQUALS(path[197769].i, 0);
        TS_ASSERT_DIFFERS(&cities[0], &path[0]);
    }
};
