#include <cmath>

typedef struct {
    double x;
    double y;
} Coord;

typedef struct {
    int i;
    bool p;
    Coord xy;
} City;

double distance_l1(Coord a, Coord b) {
    return abs(a.x-b.x) + abs(a.y-b.y);
}

double distance_l2(Coord a, Coord b) {
    return sqrt(pow(a.x-b.x, 2) + pow(a.y-b.y, 2));
}
