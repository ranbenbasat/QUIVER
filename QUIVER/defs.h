#ifndef DEFS_H
#define DEFS_H

#define CEIL(x) ((int) (x + 0.9999999999999));

#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>
#include <limits>
#include <list>
#include <chrono>
#include <functional>
#include <numeric>

using namespace std;

double sq_vnmse(vector<double>& svec, vector<double>& sqv, vector<double> *W);


#endif // !DEFS_H

