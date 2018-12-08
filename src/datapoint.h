#ifndef DATAPOINT_H
#define DATAPOINT_H
#include <cmath>

struct DataPoint {
    int _ind;
    const double* _x;
    int _D;
 
    DataPoint() : _ind(-1), _x(NULL), _D(0) {}
    DataPoint(int D, int ind, double* x) : _ind(ind), _x(x), _D(D) {}
    int index() const { return _ind; }
};

static inline double euclidean_distance(const DataPoint &t1, const DataPoint &t2) {
    double dd = 0;
    const double* x1=t1._x, *x2=t2._x;
    for(int d = 0; d < t1._D; ++d, ++x1, ++x2) {
        double tmp=*x1 - *x2;
        dd += tmp * tmp;
    }
    return std::sqrt(dd);
}

static inline double precomputed_distance(const DataPoint &t1, const DataPoint &t2) {
    // '_x' is assumed to hold a column/row of a distance matrix for 't1'.
    return t1._x[t2._ind];
}

#endif
