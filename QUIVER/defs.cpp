#include "defs.h"

#include <cstdint>
#include <iostream>
#include <iomanip>



double sq_mse(vector<double> svec, vector<double> sqv, vector<double>* W = nullptr)
{
	int curr_sqv_index = 0;
	double mse = 0;

	
	for (int i = 0; i < svec.size(); ++i)
	{
		while (svec[i] > sqv[curr_sqv_index + 1])
		{
			curr_sqv_index++;
		}
		double w = W ? (*W)[i] : 1;
		mse += (svec[i] - sqv[curr_sqv_index]) * (sqv[curr_sqv_index + 1] - svec[i]) * w;
	}

	return mse;
}




double sq_vnmse(vector<double>& svec, vector<double>& sqv, vector<double> *W = nullptr)
{
	double snorm = 0;

	if (W) {
		for (int i = 0; i < svec.size(); ++i)
		{
			snorm += svec[i] * svec[i] * (*W)[i];
		}
	}
	else {
		for (int i = 0; i < svec.size(); ++i)
		{
			snorm += svec[i] * svec[i];
		}
	}

	return sq_mse(svec, sqv, W) / snorm;
}





