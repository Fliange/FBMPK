#include <math.h>
#include <iostream>
#include "ABMC_MPK.h"

#define VALUE_TYPE double

using namespace std;

int check(VALUE_TYPE *Y1, VALUE_TYPE *Y2, int nrow){
    const double EPS = 1e-6;

    for(int i = 0; i < nrow; i++){
        if(fabs(Y1[i] - Y2[i])>EPS && fabs(Y1[i] - Y2[i])/fabs(Y1[i]) > EPS){
            cout << i << ":" << endl;
            cout << Y1[i] << endl;
            cout << Y2[i] << endl;
            cout << "\tresult error!" << endl;
            return 1;
        }
    }
    return 0;
}



int check(VALUE_TYPE* Y, VALUE_TYPE* Y_after_reorder, int nrow, ABMC_info info)
{
    int* perm = info.perm;

    double* Y_before_reorder = new double[nrow];
    for(int i = 0; i < nrow; i++){
        Y_before_reorder[i] = Y_after_reorder[perm[i]];
    }

    const double EPS = 1e-6;

    for(int i = 0; i < nrow; i++){
        if(fabs(Y[i] - Y_before_reorder[i])>EPS && fabs(Y[i] - Y_before_reorder[i])/fabs(Y[i]) > EPS){
            cout << i << ":" << endl;
            cout.precision(64);
            cout << Y[i] << endl;
            cout << Y_before_reorder[i] << endl;
            cout << "\tresult error!" << endl;
            return 1;
        }
    }
    cout << "\tresult check correct" << endl;
    return 0;
}
