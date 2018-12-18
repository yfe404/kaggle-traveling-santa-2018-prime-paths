#include <cxxtest/TestSuite.h>
#include "../src/io.hpp"

class TestIO : public CxxTest::TestSuite {
    public:
    void testReadCities(void) {
        vector<City> cities = read_cities("../test/cities.csv");
        TS_ASSERT_EQUALS(cities.size(), 197769);
    }
};
