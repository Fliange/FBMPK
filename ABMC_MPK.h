
#ifndef ABMC_MPK_H
#define ABMC_MPK_H

#include <string.h>
#include <string>
#include <math.h>
//#include <numa.h>
#include <sys/stat.h>
#include <omp.h>

#include "data_io.h"
#define VALUE_TYPE double

typedef struct ABMC_info
{
    int* colors_ptr;
    int* blocks_ptr;
    int* perm;
    int COLORS;
    int power_k;
    int nthreads;
} ABMC_info;



void ABMC_baseline(csr_mtx* mtx, ABMC_info info, VALUE_TYPE *X, VALUE_TYPE *Y_baseline)
{
    int nrow = mtx->nrow;
	int* row_ptr = mtx->row_ptr;
	int* col_ind = mtx->col_ind;
    VALUE_TYPE* val = mtx->values;
    int nthreads = info.nthreads;
    int power_k = info.power_k;

    for(int k = 0; k < power_k; k++){
#pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
        for(int i = 0; i < nrow; i++){
            VALUE_TYPE sum = 0.0;
#ifdef KAHAN
            VALUE_TYPE ac = 0.0;//kahan
#endif
            for(int j = row_ptr[i]; j < row_ptr[i + 1]; j++){
#ifdef KAHAN
                VALUE_TYPE input = val[j] * X[col_ind[j]];
                VALUE_TYPE t = sum + input;
                if(fabs(sum) >= fabs(input)){
                    ac += (sum - t) + input;
                }
                else{
                    ac += (input - t) + sum;
                }
                sum = t;
#else
                sum += val[j] * X[col_ind[j]];
#endif
            }
#ifdef KAHAN
            Y_baseline[i] = sum + ac;
#else
            Y_baseline[i] = sum;
#endif
        }
#pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
        for(int i = 0; i < nrow; i++){
            X[i] = Y_baseline[i];
        }
    }
    return;
}




void ABMC_fb_even(csr_mtx* L_mtx, csr_mtx* U_mtx, VALUE_TYPE* dia_values, VALUE_TYPE* xy, ABMC_info info, VALUE_TYPE* Y_fb)
{
    int nrow = U_mtx->nrow;
    int power_k = info.power_k;
    int COLORS = info.COLORS;
    int nthreads = info.nthreads;
    int* blocks_ptr = info.blocks_ptr;
    int* colors_ptr = info.colors_ptr;

    VALUE_TYPE* LUvec = new VALUE_TYPE[nrow];
    
#pragma omp parallel for num_threads(nthreads) 
    for(int i = 0; i < nrow; i++){
        VALUE_TYPE sumU = 0.0;
#ifdef KAHAN
        VALUE_TYPE ac = 0.0;
#endif
        for(int j = U_mtx->row_ptr[i]; j < U_mtx->row_ptr[i + 1]; j++){
#ifdef KAHAN
            VALUE_TYPE input = U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2];
            VALUE_TYPE t = sumU + input;
            if(fabs(sumU) >= fabs(input)){
                ac += (sumU - t) + input;
            }
            else{
                ac += (input - t) + sumU;
            }
            sumU = t;
#else 
            sumU += U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2];
#endif
        }
#ifdef KAHAN 
        LUvec[i] = sumU + ac;
#else
        LUvec[i] = sumU;
#endif
    }

    int k = 2;
    while(k < power_k){
        //L + D
        for(int c = 0; c < COLORS; c++){
#pragma omp parallel for num_threads(nthreads) 
            for(int blk = colors_ptr[c]; blk < colors_ptr[c + 1]; blk++){
                for(int i = blocks_ptr[blk]; i < blocks_ptr[blk + 1]; i++){
                    VALUE_TYPE sum0 = 0.0;
                    VALUE_TYPE sum1 = 0.0;
#ifdef KAHAN
                    VALUE_TYPE ac0 = 0.0;
                    VALUE_TYPE ac1 = 0.0;
#endif
                    for(int j = L_mtx->row_ptr[i]; j < L_mtx->row_ptr[i + 1]; j++){
#ifdef KAHAN
                        VALUE_TYPE input0 = L_mtx->values[j] * xy[L_mtx->col_ind[j] * 2];
                        VALUE_TYPE t0 = sum0 + input0;
                        if(fabs(sum0) >= fabs(input0)){
                            ac0 += (sum0 - t0) + input0;
                        }
                        else{
                            ac0 += (input0 - t0) + sum0;
                        }
                        sum0 = t0;

                        VALUE_TYPE input1 = L_mtx->values[j] * xy[L_mtx->col_ind[j] * 2 + 1];
                        VALUE_TYPE t1 = sum1 + input1;
                        if(fabs(sum1) >= fabs(input1)){
                            ac1 += (sum1 - t1) + input1;
                        }
                        else{
                            ac1 += (input1 - t1) + sum1;
                        }
                        sum1 = t1;
#else
                        sum0 += L_mtx->values[j] * xy[L_mtx->col_ind[j] * 2];
                        sum1 += L_mtx->values[j] * xy[L_mtx->col_ind[j] * 2 + 1];
#endif
                    }
#ifdef KAHAN
                    xy[i * 2 + 1] = sum0 + LUvec[i] + dia_values[i] * xy[i * 2] + ac0;
                    LUvec[i] = sum1 + dia_values[i] * xy[i * 2 + 1] + ac1;
#else
                    xy[i * 2 + 1] = sum0 + LUvec[i] + dia_values[i] * xy[i * 2];
                    LUvec[i] = sum1 + dia_values[i] * xy[i * 2 + 1];
#endif
                }
            }
        }

        //U
        for(int c = COLORS - 1; c >= 0; c--){
#pragma omp parallel for num_threads(nthreads) 
            for(int blk = colors_ptr[c + 1] - 1; blk >= colors_ptr[c]; blk--){
                for(int i = blocks_ptr[blk + 1] - 1; i >= blocks_ptr[blk]; i--){
                    VALUE_TYPE sum0 = 0.0;
                    VALUE_TYPE sum1 = 0.0;
#ifdef KAHAN
                    VALUE_TYPE ac0 = 0.0;
                    VALUE_TYPE ac1 = 0.0;
#endif
                    for(int j = U_mtx->row_ptr[i + 1] - 1; j >= U_mtx->row_ptr[i]; j--){
                        //sum0 += U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2 + 1];
#ifdef KAHAN
                        VALUE_TYPE input0 = U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2 + 1];
                        VALUE_TYPE t0 = sum0 + input0;
                        if(fabs(sum0) >= fabs(input0)){
                            ac0 += (sum0 - t0) + input0;
                        }
                        else{
                            ac0 += (input0 - t0) + sum0;
                        }
                        sum0 = t0;

                        //sum1 += U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2];
                        VALUE_TYPE input1 = U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2];
                        VALUE_TYPE t1 = sum1 + input1;
                        if(fabs(sum1) >= fabs(input1)){
                            ac1 += (sum1 - t1) + input1;
                        }
                        else{
                            ac1 += (input1 - t1) + sum1;
                        }
                        sum1 = t1;
#else
                        sum0 += U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2 + 1];
                        sum1 += U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2];
#endif
                    }
#ifdef KAHAN
                    xy[i * 2] = sum0 + LUvec[i] + ac0;
                    LUvec[i] = sum1 + ac1;
#else
                    xy[i * 2] = sum0 + LUvec[i];
                    LUvec[i] = sum1;
#endif
                }
            }
        }
        k+=2;
    }

	//L + D
    for(int c = 0; c < COLORS; c++){
#pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
        for(int blk = colors_ptr[c]; blk < colors_ptr[c + 1]; blk++){
            for(int i = blocks_ptr[blk]; i < blocks_ptr[blk + 1]; i++){
                VALUE_TYPE sum0 = 0.0;
                VALUE_TYPE sum1 = 0.0;
#ifdef KAHAN
                VALUE_TYPE ac0 = 0.0;
                VALUE_TYPE ac1 = 0.0;
#endif
                for(int j = L_mtx->row_ptr[i]; j < L_mtx->row_ptr[i + 1]; j++){
#ifdef KAHAN
                    VALUE_TYPE input0 = L_mtx->values[j] * xy[L_mtx->col_ind[j] * 2];
                    VALUE_TYPE t0 = sum0 + input0;
                    if(fabs(sum0) >= fabs(input0)){
                        ac0 += (sum0 - t0) + input0;
                    }
                    else{
                        ac0 += (input0 - t0) + sum0;
                    }
                    sum0 = t0;

                    VALUE_TYPE input1 = L_mtx->values[j] * xy[L_mtx->col_ind[j] * 2 + 1];
                    VALUE_TYPE t1 = sum1 + input1;
                    if(fabs(sum1) >= fabs(input1)){
                        ac1 += (sum1 - t1) + input1;
                    }
                    else{
                        ac1 += (input1 - t1) + sum1;
                    }
                    sum1 = t1;
#else
                    sum0 += L_mtx->values[j] * xy[L_mtx->col_ind[j] * 2];
                    sum1 += L_mtx->values[j] * xy[L_mtx->col_ind[j] * 2 + 1];
#endif
                }
#ifdef KAHAN
                xy[i * 2 + 1] = sum0 + LUvec[i] + dia_values[i] * xy[i * 2] + ac0;
                LUvec[i] = sum1 + dia_values[i] * xy[i * 2 + 1] + ac1;
#else
                xy[i * 2 + 1] = sum0 + LUvec[i] + dia_values[i] * xy[i * 2];
                LUvec[i] = sum1 + dia_values[i] * xy[i * 2 + 1];
#endif
            }
        }
	}
    
#pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
    for(int i = 0; i < nrow; i++){
        VALUE_TYPE sumU = 0.0;
#ifdef KAHAN
        VALUE_TYPE ac = 0.0;
#endif
        for(int j = U_mtx->row_ptr[i]; j < U_mtx->row_ptr[i + 1]; j++){
#ifdef KAHAN
            VALUE_TYPE input = U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2 + 1];
            VALUE_TYPE t = sumU + input;
            if(fabs(sumU) >= fabs(input)){
                ac += (sumU - t) + input;
            }
            else{
                ac += (input - t) + sumU;
            }
            sumU = t;
#else
            sumU += U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2 + 1];
#endif
        }
        //Y_fb[i] = sumU / info.max[power_k - 1];
#ifdef KAHAN
        Y_fb[i] = sumU + LUvec[i] + ac;
#else
        Y_fb[i] = sumU + LUvec[i];
#endif
    }

    return;
}


void ABMC_fb_odd(csr_mtx* L_mtx, csr_mtx* U_mtx, VALUE_TYPE* dia_values, VALUE_TYPE* xy, ABMC_info info, VALUE_TYPE* Y_fb)
{
    int nrow = L_mtx->nrow;
    int power_k = info.power_k;
    int COLORS = info.COLORS;
    int nthreads = info.nthreads;
    int* blocks_ptr = info.blocks_ptr;
    int* colors_ptr = info.colors_ptr;

    VALUE_TYPE* LUvec = new VALUE_TYPE[nrow];
    
#pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
    for(int i = 0; i < nrow; i++){
        VALUE_TYPE sumU = 0.0;
#ifdef KAHAN
        VALUE_TYPE ac = 0.0;
#endif
        for(int j = U_mtx->row_ptr[i]; j < U_mtx->row_ptr[i + 1]; j++){
#ifdef KAHAN
            VALUE_TYPE input = U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2];
            VALUE_TYPE t = sumU + input;
            if(fabs(sumU) >= fabs(input)){
                ac += (sumU - t) + input;
            }
            else{
                ac += (input - t) + sumU;
            }
            sumU = t;
#else
            sumU += U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2];
#endif
        }
#ifdef KAHAN
        LUvec[i] = sumU + ac;
#else
        LUvec[i] = sumU;
#endif
    }
    

    int k = 1;
    while(k < power_k){
        //L + D
        for(int c = 0; c < COLORS; c++){
#pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
            for(int blk = colors_ptr[c]; blk < colors_ptr[c + 1]; blk++){
                for(int i = blocks_ptr[blk]; i < blocks_ptr[blk + 1]; i++){
                    double sum0 = 0.0;
                    double sum1 = 0.0;
#ifdef KAHAN
                    VALUE_TYPE ac0 = 0.0;
                    VALUE_TYPE ac1 = 0.0;
#endif
                    for(int j = L_mtx->row_ptr[i]; j < L_mtx->row_ptr[i + 1]; j++){
#ifdef KAHAN
                        VALUE_TYPE input0 = L_mtx->values[j] * xy[L_mtx->col_ind[j] * 2];
                        VALUE_TYPE t0 = sum0 + input0;
                        if(fabs(sum0) >= fabs(input0)){
                            ac0 += (sum0 - t0) + input0;
                        }
                        else{
                            ac0 += (input0 - t0) + sum0;
                        }
                        sum0 = t0;

                        VALUE_TYPE input1 = L_mtx->values[j] * xy[L_mtx->col_ind[j] * 2 + 1];
                        VALUE_TYPE t1 = sum1 + input1;
                        if(fabs(sum1) >= fabs(input1)){
                            ac1 += (sum1 - t1) + input1;
                        }
                        else{
                            ac1 += (input1 - t1) + sum1;
                        }
                        sum1 = t1;
#else
                        sum0 += L_mtx->values[j] * xy[L_mtx->col_ind[j] * 2];
                        sum1 += L_mtx->values[j] * xy[L_mtx->col_ind[j] * 2 + 1];
#endif
                    }
                    //xy[i * 2 + 1] = sum0 / info.max[k - 1];
#ifdef KAHAN
                    xy[i * 2 + 1] = sum0 + LUvec[i] + dia_values[i] * xy[i * 2] + ac0;
                    LUvec[i] = sum1 + dia_values[i] * xy[i * 2 + 1] + ac1;
#else
                    xy[i * 2 + 1] = sum0 + LUvec[i] + dia_values[i] * xy[i * 2];
                    LUvec[i] = sum1 + dia_values[i] * xy[i * 2 + 1];
#endif
                }
            }
        }

        //U
        for(int c = COLORS - 1; c >= 0; c--){
#pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
            for(int blk = colors_ptr[c + 1] - 1; blk >= colors_ptr[c]; blk--){
                for(int i = blocks_ptr[blk + 1] - 1; i >= blocks_ptr[blk]; i--){
                    VALUE_TYPE sum0 = 0.0;
                    VALUE_TYPE sum1 = 0.0;
#ifdef KAHAN
                    VALUE_TYPE ac0 = 0.0;
                    VALUE_TYPE ac1 = 0.0;
#endif
                    for(int j = U_mtx->row_ptr[i + 1] - 1; j >= U_mtx->row_ptr[i]; j--){
#ifdef KAHAN
                        VALUE_TYPE input0 = U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2 + 1];
                        VALUE_TYPE t0 = sum0 + input0;
                        if(fabs(sum0) >= fabs(input0)){
                            ac0 += (sum0 - t0) + input0;
                        }
                        else{
                            ac0 += (input0 - t0) + sum0;
                        }
                        sum0 = t0;

                        VALUE_TYPE input1 = U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2];
                        VALUE_TYPE t1 = sum1 + input1;
                        if(fabs(sum1) >= fabs(input1)){
                            ac1 += (sum1 - t1) + input1;
                        }
                        else{
                            ac1 += (input1 - t1) + sum1;
                        }
                        sum1 = t1;
#else
                        sum0 += U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2 + 1];
                        sum1 += U_mtx->values[j] * xy[U_mtx->col_ind[j] * 2];
#endif
                    }
#ifdef KAHAN
                    xy[i * 2] = sum0 + LUvec[i] + ac0;
                    LUvec[i] = sum1 + ac1;
#else
                    xy[i * 2] = sum0 + LUvec[i];
                    LUvec[i] = sum1;
#endif
                }
            }
        }

        k+=2;
    }

#pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
    for(int i = 0; i < nrow; i++){
        VALUE_TYPE sumL = 0.0;
#ifdef KAHAN
        VALUE_TYPE ac = 0.0;
#endif
        for(int j = L_mtx->row_ptr[i]; j < L_mtx->row_ptr[i + 1]; j++){
#ifdef KAHAN
            VALUE_TYPE input = L_mtx->values[j] * xy[L_mtx->col_ind[j] * 2];
            VALUE_TYPE t = sumL + input;
            if(fabs(sumL) >= fabs(input)){
                ac += (sumL - t) + input;
            }
            else{
                ac += (input - t) + sumL;
            }
            sumL = t;
#else
            sumL += L_mtx->values[j] * xy[L_mtx->col_ind[j] * 2];           
#endif
        }
#ifdef KAHAN
        Y_fb[i] = sumL + LUvec[i] + dia_values[i] * xy[i * 2] + ac;
#else
        Y_fb[i] = sumL + LUvec[i] + dia_values[i] * xy[i * 2];
#endif
    }

    return;
}


int ABMC_fb(csr_mtx* L_mtx, csr_mtx* U_mtx, VALUE_TYPE* dia_values, VALUE_TYPE* xy, ABMC_info info, VALUE_TYPE *Y_fb)
{
    if(info.power_k % 2 == 1){
        ABMC_fb_odd(L_mtx, U_mtx, dia_values, xy, info, Y_fb);
    }
    else{
        ABMC_fb_even(L_mtx, U_mtx, dia_values, xy, info, Y_fb);
    }
    return 0;
}

#endif