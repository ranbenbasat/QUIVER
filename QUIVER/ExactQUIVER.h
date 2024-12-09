#pragma once
#ifndef EXACT_QUIVER_H
#define EXACT_QUIVER_H

#include "defs.h"

// A class that implements the (exact) QUIVER algorithm
// It supports three variants: (1) unweighted and unaccelerated (2) weighted and unaccelerated (3) unweighted and accelerated.
template <bool is_weighted, bool is_accelerated>
class ExactQUIVER {
    double* cumsum0; // sum cumsum[i] = \sum_{j=0}^i sorted_vec_weights[j]                                      (unused if is_weighted == false)
    double* cumsum1; // sum cumsum[i] = \sum_{j=0}^i sorted_vec_weights[j] * sorted_vec[j]                      (if is_weighted == false then sorted_vec_weights[j] = 1)
    double* cumsum2; // sum cumsum[i] = \sum_{j=0}^i sorted_vec_weights[j] * sorted_vec[j]^2                    (if is_weighted == false then sorted_vec_weights[j] = 1)
    const double* const sorted_vec;         // input
    const double* const sorted_vec_weights; // input
    int d;                                  // vector dimension
    int s_d;                                // current matrix offset
    int s_minus_1__d;                       // prev layer offset

    uint32_t* rows;                         // Used by the SMAWK algorithm
    uint32_t* cols;                         // Used by the SMAWK algorithm
    int* col_idx_lookup;                    // Used by the SMAWK algorithm

    double* D;                              // The cost values of the dynamic program
    uint32_t* T;                            // The argmin of the dynamic program (the index that minimizes D)

public:
    /*
    * C'tor, receives as arguments the (sorted!) vector to be quantized, `sorted_vec`, its dimension, `d`, and a potential weight vector, `sorted_vec_weights`
    */
    ExactQUIVER(double* sorted_vec, uint32_t d, double* sorted_vec_weights = nullptr) : sorted_vec(sorted_vec), d(d), sorted_vec_weights(sorted_vec_weights) {
        if (is_weighted && is_accelerated) {
            throw runtime_error("is_weighted && is_accelerated is not supported\n");
        }
        cumsum1 = new double[d]();
        cumsum2 = new double[d]();
        if (is_weighted) {
            cumsum0 = new double[d]();
        }
        preprocess();
    }

    ~ExactQUIVER() {
        delete[] cumsum1;
        delete[] cumsum2;
        if (is_weighted) {
            delete[] cumsum0;
        }
    }

    /*
    * A wrapper for the SMAWK algorithm (Aggarwal, Alok; Klawe, Maria M.; Moran, Shlomo; Shor, Peter; Wilber, Robert (1987), "Geometric applications of a matrix-searching algorithm", Algorithmica, 2 (2): 195–208, doi:10.1007/BF01840359)
    * This is an optimized version of the code by https://github.com/dstein64/kmeans1d/blob/master/kmeans1d/_core.cpp by Daniel Steinberg
    */
    void smawk(
        const uint32_t num_rows,
        const uint32_t num_cols,
        uint32_t s_) {
        s_d = s_ * d;
        s_minus_1__d = s_d - d;
        iota(rows, rows + num_rows, 0);
        iota(cols, cols + num_cols, 0);
        _smawk(rows, cols, num_rows, num_cols, s_);
        return;
    }

    /*
    * The recursive function of the SMAWK algorithm (Aggarwal, Alok; Klawe, Maria M.; Moran, Shlomo; Shor, Peter; Wilber, Robert (1987), "Geometric applications of a matrix-searching algorithm", Algorithmica, 2 (2): 195–208, doi:10.1007/BF01840359)
    * This is an optimized version of the code by https://github.com/dstein64/kmeans1d/blob/master/kmeans1d/_core.cpp by Daniel Steinberg
    */
    void _smawk(
        const uint32_t* rows,
        const uint32_t* cols,
        int rows_size,
        int cols_size,
        uint32_t s_) {

        if (rows_size == 0) return;

        uint32_t* _cols = new uint32_t[cols_size];
        int _cols_size = 0;
        for (int col_idx = 0; col_idx < cols_size; ++col_idx) {
            uint32_t col = cols[col_idx];
            while (true) {
                if (_cols_size == 0) break;
                uint32_t row = rows[_cols_size - 1];
                if (is_accelerated) {
                    if (lookup2(row, col) >= lookup2(row, _cols[_cols_size - 1]))
                        break;
                }
                else {
                    if (lookup(row, col) >= lookup(row, _cols[_cols_size - 1]))
                        break;
                }
                _cols_size--;
            }
            if (_cols_size < rows_size)
                _cols[_cols_size++] = col;
        }

        uint32_t* odd_rows = new uint32_t[rows_size / 2 + 1];
        int odd_rows_size = 0;
        for (int i = 1; i < rows_size; i += 2) {
            odd_rows[odd_rows_size++] = rows[i];
        }
        _smawk(odd_rows, _cols, odd_rows_size, _cols_size, s_);

        for (int idx = 0; idx < _cols_size; ++idx) {
            col_idx_lookup[_cols[idx]] = idx;
        }

        uint32_t start = 0;
        for (int r = 0; r < rows_size; r += 2) {
            uint32_t row = rows[r];
            uint32_t stop = _cols_size - 1;
            if (r < rows_size - 1)
                stop = col_idx_lookup[T[s_d + rows[r + 1]]];
            uint32_t argmin = _cols[start];
            double min;
            if (is_accelerated) {
                min = lookup2(row, argmin);
            }
            else {
                min = lookup(row, argmin);
            }
            for (int c = start + 1; c <= stop; ++c) {
                double value;
                if (is_accelerated) {
                    value = lookup2(row, _cols[c]);
                }
                else {
                    value = lookup(row, _cols[c]);
                }
                if (value < min) {
                    argmin = _cols[c];
                    min = value;
                }
            }
            int loc = s_d + row;
            T[loc] = argmin;
            D[loc] = min;
            start = stop;
        }

        delete[] _cols;
        delete[] odd_rows;
    }

    /*
    * The main function. Receives as input the desired number of quantization values, `s`, and returns a vector with the optimal quantization values.
    */
    vector<double> calcQuantizationValues(
        uint32_t s) {

        vector<double> quant_values(s);
        quant_values[0] = sorted_vec[0];
        quant_values[s - 1] = sorted_vec[d - 1];

        if (s == 2) {
            return quant_values;
        }

        if (is_accelerated) {
            D = new double[s / 2 * d]();
            T = new uint32_t[s / 2 * d]();
        }
        else {
            D = new double[(s - 1) * d]();
            T = new uint32_t[(s - 1) * d]();
        }

        if (is_accelerated) {
            if (s % 2) {
                for (uint32_t i = 3; i < d; ++i) {
                    D[i] = calc2(0, i);

                }
            }
            else {
                for (uint32_t i = 2; i < d; ++i) {
                    D[i] = calc(0, i);
                }
            }
        }
        else {
            for (uint32_t i = 2; i < d; ++i) {
                D[i] = calc(0, i);
            }
        }

        col_idx_lookup = new int[d]();      // Used by the SMAWK algorithm
        rows = new uint32_t[d];             // Used by the SMAWK algorithm
        cols = new uint32_t[d];             // Used by the SMAWK algorithm

        if (is_accelerated) {
            for (uint32_t s_ = 1; s_ < s / 2; ++s_) {
                smawk(d, d, s_);
            }
        }
        else {
            for (uint32_t s_ = 1; s_ < s - 1; ++s_) {
                smawk(d, d, s_);
            }
        }

        delete[] col_idx_lookup;
        delete[] D;

        delete[] rows;
        delete[] cols;

        int idx = d - 1;

        if (is_accelerated) { // Interpolate the optimal solution by filling in every other quantization point
            s_d = (s / 2 - 1) * d;
            int idx = d - 1;
            for (int j = 1; j < s / 2; ++j) {
                int new_idx = T[s_d + idx];

                auto a = sorted_vec[new_idx];
                auto c = sorted_vec[idx];

                double float_b = (c * idx - a * new_idx - (cumsum1[idx] - cumsum1[new_idx])) / (c - a);
                int b = CEIL(float_b);
                quant_values[s - 1 - 2 * j] = sorted_vec[new_idx];
                quant_values[s - 2 * j] = sorted_vec[b];
                idx = new_idx;
                s_d -= d;
            }
            delete[] T;

            if (s % 2) {
                auto a = sorted_vec[0];
                auto c = sorted_vec[idx];
                double float_b = (c * idx - (cumsum1[idx] - cumsum1[0])) / (c - a);
                int b = CEIL(float_b);
                quant_values[1] = sorted_vec[b];
            }
        }
        else {
            s_d = (s - 2) * d;
            for (int j = 1; j < s - 1; ++j) {
                int new_idx = T[s_d + idx];
                quant_values[s - 1 - j] = sorted_vec[new_idx];
                idx = new_idx;
                s_d -= d;
            }
            delete[] T;
        }
        return quant_values;
    }
private:

    // Calculate the sum of variances of all points between either point k and point j, assuming two quantization points at k and j
    inline double calc(int k, int j) const {
        if (j > k + 1) {
            double sum_of_weights;
            if (is_weighted)
                sum_of_weights = cumsum0[j] - cumsum0[k];
            else
                sum_of_weights = (j - k);
            return -sorted_vec[j] * sorted_vec[k] * sum_of_weights + (sorted_vec[j] + sorted_vec[k]) * (cumsum1[j] - cumsum1[k]) - (cumsum2[j] - cumsum2[k]);
        }
        return 0;
    }

    // Calculate the sum of variances of all points between either point k and point j, assuming two quantization points at k and j and a third placed optimally in between
    inline double calc2(int k, int j) const {
        if (j > k + 2) {
            auto a = sorted_vec[k];
            auto c = sorted_vec[j];

            double float_b = (c * j - a * k - (cumsum1[j] - cumsum1[k])) / (c - a);
            int b =  CEIL(float_b);
            double res = float_b == float_b ? -(cumsum2[j] - cumsum2[k]) + (sorted_vec[b] + sorted_vec[k]) * (cumsum1[b] - cumsum1[k]) + (sorted_vec[b] + sorted_vec[j]) * (cumsum1[j] - cumsum1[b]) - sorted_vec[b] * (sorted_vec[k] * (b - k) + sorted_vec[j] * (j - b)) : 0;
            return res;
        }
        return 0;
    }

    // Calculate MSE(row-1, col) + calc(col, row)
    inline double lookup(uint32_t row, uint32_t col) const {
        return D[s_minus_1__d + col] + calc(col, row);
    }
    // Calculate MSE(row-2, col) + calc2(col, row)
    inline double lookup2(uint32_t row, uint32_t col) const {
        return D[s_minus_1__d + col] + calc2(col, row);
    }

    // Compute the cumulative sums that are used for computing calc(2) in constant time
    void preprocess() {
        if (is_weighted) {
            cumsum0[0] = sorted_vec_weights[0];
            cumsum1[0] = sorted_vec[0] * sorted_vec_weights[0];
            cumsum2[0] = sorted_vec[0] * sorted_vec[0] * sorted_vec_weights[0];
            for (uint32_t i = 1; i < d; ++i) {
                double x = sorted_vec[i];
                double curw = sorted_vec_weights[i];
                cumsum0[i] = (curw + cumsum0[i - 1]);
                cumsum1[i] = (curw * x + cumsum1[i - 1]);
                cumsum2[i] = (curw * x * x + cumsum2[i - 1]);
            }
        }
        else {
            cumsum1[0] = sorted_vec[0];
            cumsum2[0] = sorted_vec[0] * sorted_vec[0];
            for (uint32_t i = 1; i < d; ++i) {
                double x = sorted_vec[i];
                cumsum1[i] = x + cumsum1[i - 1];
                cumsum2[i] = x * x + cumsum2[i - 1];
            }
        }
    }
};

#endif // !EXACT_QUIVER_H
