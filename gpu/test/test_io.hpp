#include <cxxtest/TestSuite.h>
// #include <filesystem>
// ^ Requires GCC 8, but CUDA supports only GCC 7,
//   so we cannot use temp_directory_path().
#include "../src/io.hpp"

using namespace std;

class TestIO : public CxxTest::TestSuite {
    public:
    void testReadCities(void) {
        auto cities = read_cities<double>("../test/cities.csv");
        TS_ASSERT_EQUALS(cities.size(), 197769);
        TS_ASSERT_EQUALS(cities[2].i, 2);
        TS_ASSERT_EQUALS(cities[2].p, true);
        TS_ASSERT_EQUALS(cities[2].xy.x, 3454.15819771172);
        TS_ASSERT_EQUALS(cities[2].xy.y, 2820.05301124811);
        TS_ASSERT_EQUALS(cities[197768].i, 197768);
        TS_ASSERT_EQUALS(cities[197768].p, false);
    }

    void testReadPath(void) {
        auto cities = read_cities<double>("../test/cities.csv");
        auto path = read_path(cities, "../test/1516773.csv");
        TS_ASSERT_EQUALS(path.size(), 197770);
        TS_ASSERT_EQUALS(path[0].i, 0);
        TS_ASSERT_EQUALS(path[197769].i, 0);
        TS_ASSERT_DIFFERS(&cities[0], &path[0]);
        TS_ASSERT_EQUALS(score(path), 1516773.9447755208);
        TS_ASSERT(is_valid(path.begin(), path.end()));
    }

    void testWritePath(void) {
        auto cities = read_cities<double>("../test/cities.csv");
        auto path = read_path(cities, "../test/1516773.csv");
        string fp = "path.tmp";
        write_path(path.begin(), path.end(), fp);
        auto new_path = read_path(cities, fp);
        TS_ASSERT_EQUALS(path.size(), new_path.size());
        for (size_t i = 0; i < path.size(); i++) {
            TS_ASSERT_EQUALS(path[i].i, new_path[i].i);
        }
    }
};
