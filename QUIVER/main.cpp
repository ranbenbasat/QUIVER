#include <stdio.h>              // Needed for printf()
#include <stdlib.h>             // Needed for exit() and ato*()
#include <fstream>
#include "defs.h"
#include "ApproxQUIVER.h"
#include "ExactQUIVER.h"
#include <random>



int main()
{
    {
        default_random_engine generator;
        generator.seed(42);

        
        int n = 1<<17;

        int s = 16;
        int M = 100000;

        bool run_ExactQUIVER = true;
        bool run_WeightedExactQUIVER = true;
        bool run_AccelQUIVER = true;

        bool run_ApproxQUIVER = true;
        bool run_WeightedApproxQUIVER = true;

        auto start = chrono::high_resolution_clock::now();
        auto stop = chrono::high_resolution_clock::now();


        //lognormal_distribution<double> distribution(0.0, 1);
        normal_distribution<double> distribution(0.0, 1);
        uniform_real_distribution<double> unidistribution(0.0, 1);
        vector<double> vec(n);
        vector<double> svec;
        int idx = -1;
        double norm = 0;

        for (int i = 0; i < n; ++i) {
            double number = distribution(generator);
            norm += number * number;
            vec[++idx] = number;
        }
        vec.resize(idx + 1);
        svec.resize(idx + 1);
        cout << "norm = " << norm << endl;           


        start = chrono::high_resolution_clock::now();
        partial_sort_copy(vec.begin(), vec.end(), svec.begin(), svec.end());
        stop = chrono::high_resolution_clock::now();    
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "partial_sort_copy time: " << duration.count() / 1000 << " ms" << endl;


        bool debug = true;

        vector<double> resmoke;

        double eps = 1;

        double vnmse;

        vector<double> W(n, 1);
        for (int i = 0; i < n; ++i) {
			W[i] = 1; // to compare with unweighted version
        }
        if (run_ExactQUIVER) {
            start = chrono::high_resolution_clock::now();
            ExactQUIVER<false, false> eq(svec.data(), (uint32_t)n, nullptr);
            auto quant_values = eq.calcQuantizationValues(s);
            stop = chrono::high_resolution_clock::now();

            vnmse = sq_vnmse(svec, quant_values, nullptr);

            cout << "ExactQUIVER: ";
            cout << "[";
            for (int i = 0; i < s; ++i) {
                cout << quant_values[i] << ", ";
            }

            cout << "], vnmse = " << vnmse << endl;
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "ExactQUIVER time: " << duration.count() / 1000 << " ms" << endl;
        }
        if (run_WeightedExactQUIVER) {
            start = chrono::high_resolution_clock::now();
            ExactQUIVER<true, false> eq(svec.data(), (uint32_t)n, W.data());
            auto quant_values = eq.calcQuantizationValues(s);
            stop = chrono::high_resolution_clock::now();

            vnmse = sq_vnmse(svec, quant_values, &W);

            cout << "WeightedExactQUIVER: ";
            cout << "[";
            for (int i = 0; i < s; ++i) {
                cout << quant_values[i] << ", ";
            }

            cout << "], vnmse = " << vnmse << endl;
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "WeightedExactQUIVER time: " << duration.count() / 1000 << " ms" << endl;
        }
        if (run_AccelQUIVER) {
            start = chrono::high_resolution_clock::now();
            ExactQUIVER<false, true> aeq(svec.data(), (uint32_t)n, nullptr);
            auto quant_values = aeq.calcQuantizationValues(s);
            stop = chrono::high_resolution_clock::now();

            vnmse = sq_vnmse(svec, quant_values, nullptr);

            cout << "AccelQUIVER: ";
            cout << "[";
            for (int i = 0; i < s; ++i) {
                cout << quant_values[i] << ", ";
            }

            cout << "], vnmse = " << vnmse << endl;
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "AccelQUIVER time: " << duration.count() / 1000 << " ms" << endl;
        }
        if (run_ApproxQUIVER) {
            start = chrono::high_resolution_clock::now();
            ApproxQUIVER<false> taq(svec.data(), (uint32_t)n, nullptr, M);
            auto quant_values = taq.calcQuantizationValues(s);
            stop = chrono::high_resolution_clock::now();

            vnmse = sq_vnmse(svec, quant_values, nullptr);

            cout << "ApproxQUIVER: ";
            cout << "[";
            for (int i = 0; i < s; ++i) {
                cout << quant_values[i] << ", ";
            }

            cout << "], vnmse = " << vnmse << endl;
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "ApproxQUIVER time: " << duration.count() / 1000 << " ms" << endl;
        }
        if (run_WeightedApproxQUIVER) {
            start = chrono::high_resolution_clock::now();
            ApproxQUIVER<true> taq(svec.data(), (uint32_t)n, W.data(), M);
            auto quant_values = taq.calcQuantizationValues(s);
            stop = chrono::high_resolution_clock::now();

            vnmse = sq_vnmse(svec, quant_values, &W);

            cout << "WeightedApproxQUIVER: ";
            cout << "[";
            for (int i = 0; i < s; ++i) {
                cout << quant_values[i] << ", ";
            }

            cout << "], vnmse = " << vnmse << endl;
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "WeightedApproxQUIVER time: " << duration.count() / 1000 << " ms" << endl;
        }
    }

    // Check for memory leaks: send all reports to STDOUT
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDOUT);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDOUT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDOUT);
    _CrtDumpMemoryLeaks();

    return 0;
}


