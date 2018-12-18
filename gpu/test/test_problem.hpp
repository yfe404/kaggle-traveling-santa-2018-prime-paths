#include <cxxtest/TestSuite.h>
#include "../src/problem.hpp"

class TestProblem : public CxxTest::TestSuite {
    public:
    void testDistanceL1(void) {
        Coord a = {1.5, 2.5};
        Coord b = {8.5, 4.5};
        TS_ASSERT_EQUALS(distance_l1(a, b), 9);
    }

    void testDistanceL2(void) {
        Coord a = {1.5, 2.5};
        Coord b = {8.5, 4.5};
        TS_ASSERT_DELTA(distance_l2(a, b), 7.280109889280518, 1e-12);
    }
};
