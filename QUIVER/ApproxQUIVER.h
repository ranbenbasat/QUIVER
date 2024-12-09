#pragma once
#ifndef TEMPLATE_APPROX_QUIVER_H
#define TEMPLATE_APPROX_QUIVER_H
#include "defs.h"
#include <iomanip>


// A class that implements the (approximate) QUIVER algorithm
// It supports two variants: (1) unweighted (2) weighted.
template <bool is_weighted>
class ApproxQUIVER {
    double* cumsum0;
    double* cumsum1;
    double* cumsum2;
    const double* const sorted_vec;
    const double* const sorted_vec_weights;
    int d;
    int s_M;
    int s_minus_1__M;                       // prev layer offset
    double minval, maxval;
    double delta;

    uint32_t* rows;
    uint32_t* cols;

    int* col_idx_lookup;
    int* inv_orig_hist_bin;
    

    double* D;
    uint32_t* T;

    int M;
    const int orig_M;

public:
    ApproxQUIVER(double* sorted_vec, uint32_t d, double* sorted_vec_weights = nullptr, int M = 1000) : sorted_vec(sorted_vec), d(d), sorted_vec_weights(sorted_vec_weights), M(M), orig_M(M) {
        cumsum0 = new double[M]();
        cumsum1 = new double[M]();
        cumsum2 = new double[M]();
        inv_orig_hist_bin = new int[M]();
        preprocess();
    }

    ~ApproxQUIVER() {
        delete[] cumsum0;
        delete[] cumsum1;
        delete[] cumsum2;
        delete[] inv_orig_hist_bin;
    }

    void smawk(
        const uint32_t num_rows,
        const uint32_t num_cols,
        uint32_t s_) {
        s_M = s_ * M;
        s_minus_1__M = s_M - M;
        iota(rows, rows + num_rows, 0);
        iota(cols, cols + num_cols, 0);
        _smawk(rows, cols, num_rows, num_cols, s_);
        return;
    }

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
                if (lookup(row, col, s_) >= lookup(row, _cols[_cols_size - 1], s_))
                    break;
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
                stop = col_idx_lookup[T[s_M + rows[r + 1]]];
            uint32_t argmin = _cols[start];
            double min = lookup(row, argmin, s_);
            for (int c = start + 1; c <= stop; ++c) {
                double value = lookup(row, _cols[c], s_);
                if (value < min) {
                    argmin = _cols[c];
                    min = value;
                }
            }
            int loc = s_M + row;
            T[loc] = argmin;
            D[loc] = min;
            start = stop;
        }

        delete[] _cols;
        delete[] odd_rows;
    }

    vector<double> calcQuantizationValues(
        uint32_t s) {

        D = new double[(s - 1) * M]();
        T = new uint32_t[(s - 1) * M]();

        for (uint32_t i = 1; i < M; ++i) {
            D[i] = calc(0, i);
        }

        col_idx_lookup = new int[M]();
        rows = new uint32_t[M];
        cols = new uint32_t[M];

        for (uint32_t s_ = 1; s_ < s - 1; ++s_) {
            smawk(M, M, s_);
        }

        delete[] col_idx_lookup;
        delete[] D;

        delete[] rows;
        delete[] cols;

        vector<double> quant_values(s);
        quant_values[0] = sorted_vec[0];
        quant_values[s - 1] = sorted_vec[d - 1];
        s_M = (s - 2) * M;
        int idx = M - 1;
        for (int j = 1; j < s - 1; ++j) {
            int new_idx = T[s_M + idx];
            quant_values[s - 1 - j] = minval + delta * inv_orig_hist_bin[new_idx];//sorted_vec[new_idx];
            idx = new_idx;
            s_M -= M;
        }

        delete[] T;
        return quant_values;
    }
private:

    inline double calc(int k, int j) {
        if (j > k) {
            auto sum_of_weights = cumsum0[j] - cumsum0[k];
            double jval = minval + delta * inv_orig_hist_bin[j];
            double kval = minval + delta * inv_orig_hist_bin[k];
            return -jval * kval * sum_of_weights + (jval + kval) * (cumsum1[j] - cumsum1[k]) - (cumsum2[j] - cumsum2[k]);
        }
        return 0;
    }

    double lookup(uint32_t row, uint32_t col, uint32_t s_) {
        return D[s_minus_1__M + col] + calc(col, row);
    }

    void preprocess() {
        minval = sorted_vec[0];
        maxval = sorted_vec[d - 1];

        delta = (maxval - minval) / (M - 1);

        if (is_weighted) {
            double x = sorted_vec[0];
            double curw = sorted_vec_weights[0];
            cumsum0[0] = curw;
            cumsum1[0] = curw * x;
            cumsum2[0] = curw * x * x;
            for (int i = 1; i < d; ++i) {
                double x = sorted_vec[i];
                int grindex = (x != maxval) ? ceil((x - minval) / delta) : M - 1;
                double curw = sorted_vec_weights[i];
                cumsum0[grindex] += curw;
                cumsum1[grindex] += curw * x;
                cumsum2[grindex] += curw * x * x;
            }
        }
        else {
            double x = sorted_vec[0];
            cumsum0[0] = 1;
            cumsum1[0] = x;
            cumsum2[0] = x * x;
            for (int i = 1; i < d; ++i) {
                double x = sorted_vec[i];
                int grindex = (x != maxval) ? ceil((x - minval) / delta) : M - 1;
                cumsum0[grindex] += 1;
                cumsum1[grindex] += x;
                cumsum2[grindex] += x * x;
            }
        }

        int M2 = 1; // New size after removing zeros
        for (int i = 1; i < M; ++i) {
            if (cumsum0[i] != 0) { // Check if the element is not zero
                cumsum0[M2] = cumsum0[i];
                cumsum1[M2] = cumsum1[i];
                cumsum2[M2] = cumsum2[i];
                inv_orig_hist_bin[M2] = i;
                M2++; // Increment new size
            }
        }
        M = M2;

        for (int i = 0; i < M - 1; ++i) {
            cumsum0[i + 1] += cumsum0[i];
            cumsum1[i + 1] += cumsum1[i];
            cumsum2[i + 1] += cumsum2[i];
        }
    }
};
#endif // !APPROX_QUIVER_H
