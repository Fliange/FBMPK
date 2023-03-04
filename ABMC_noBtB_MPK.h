#ifndef ABMC_NOBTB_MPK_H
#define ABMC_NOBTB_MPK_H

#include "ABMC_MPK.h"


#define VALUE_TYPE double


void copy_vector(VALUE_TYPE* X, VALUE_TYPE* Y, int size,int nthreads)
{
#pragma omp parallel for num_threads(nthreads) schedule(guided) proc_bind(spread)
    for(int i = 0; i < size; i++){
        X[i] = Y[i];
    }
}

void cmp_fb_odd(csr_mtx* L_mtx, csr_mtx* U_mtx, VALUE_TYPE* dia_values, VALUE_TYPE *X, ABMC_info info, VALUE_TYPE* Y_fb)
{
    int nrow = L_mtx->nrow;
    int power_k = info.power_k;
    int COLORS = info.COLORS;
    int nthreads = info.nthreads;
    int* blocks_ptr = info.blocks_ptr;
    int* colors_ptr = info.colors_ptr;

    VALUE_TYPE* Y = Y_fb;

    VALUE_TYPE* LUvec = new VALUE_TYPE[nrow];
    

#pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
    for(int i = 0; i < nrow; i++){
        double sumU = 0.0;
        for(int j = U_mtx->row_ptr[i]; j < U_mtx->row_ptr[i+1]; j++){
            sumU += U_mtx->values[j] * X[U_mtx->col_ind[j]];
        }
        LUvec[i] = sumU;
    }

    int k = 1;
    while(k < power_k){
        //L
        for(int c = 0; c < COLORS; c++){
#pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
            for(int blk = colors_ptr[c]; blk < colors_ptr[c + 1]; blk++){
                for(int i = blocks_ptr[blk]; i < blocks_ptr[blk + 1]; i++){
                    double sumL[2] = {0.0, 0.0};
                    for(int j = L_mtx->row_ptr[i]; j < L_mtx->row_ptr[i + 1]; j++){
                        sumL[0] += L_mtx->values[j] * X[L_mtx->col_ind[j]];
                        sumL[1] += L_mtx->values[j] * Y[L_mtx->col_ind[j]];
                    }
                    //Y[k][i] = (LUvec[i] + sumL[0] + dia_values[i] * Y[k - 1][i]) / info.max[k - 1];
                    Y[i] = (LUvec[i] + sumL[0] + dia_values[i] * X[i]);
                    LUvec[i] = sumL[1];
                }
            }
        }
        copy_vector(X, Y, nrow, nthreads);

        //U
        for(int c = COLORS - 1; c >= 0; c--){
#pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
            for(int blk = colors_ptr[c + 1] - 1; blk >= colors_ptr[c]; blk--){
                for(int i = blocks_ptr[blk + 1] - 1; i >= blocks_ptr[blk]; i--){
                    double sumU[2] = {0.0, 0.0};
                    for(int j = U_mtx->row_ptr[i + 1] - 1; j >= U_mtx->row_ptr[i]; j--){//倒序访问
                        sumU[0] += U_mtx->values[j] * X[U_mtx->col_ind[j]];
                        sumU[1] += U_mtx->values[j] * Y[U_mtx->col_ind[j]]; 
                    }
                    //Y[k + 1][i] = (LUvec[i] + sumU[0] + dia_values[i] * Y[k][i]) / info.max[k];
                    Y[i] = (LUvec[i] + sumU[0] + dia_values[i] * X[i]);
                    LUvec[i] = sumU[1];
                }
            }
        }
        copy_vector(X, Y, nrow,nthreads);
        k += 2;
    }
    
    //最后一次迭代，补L
    #pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
    for(int i = 0; i<nrow; i++){
        double sumL = 0.0;
        for(int j = L_mtx->row_ptr[i]; j < L_mtx->row_ptr[i+1]; j++){
            sumL += L_mtx->values[j] * X[L_mtx->col_ind[j]];
        }
        //Y[power_k][i] = (LUvec[i] + sumL + dia_values[i] * Y[power_k - 1][i]) / info.max[power_k - 1];
        Y_fb[i] = (LUvec[i] + sumL + dia_values[i] * X[i]);
    }
    return;
}


void cmp_fb_even(csr_mtx* L_mtx, csr_mtx* U_mtx, VALUE_TYPE* dia_values, VALUE_TYPE *X, ABMC_info info, VALUE_TYPE* Y_fb)
{
    int nrow = U_mtx->nrow;
    int power_k = info.power_k;
    int COLORS = info.COLORS;
    int nthreads = info.nthreads;
    int* blocks_ptr = info.blocks_ptr;
    int* colors_ptr = info.colors_ptr;

    VALUE_TYPE* Y = Y_fb;
    
    VALUE_TYPE* LUvec = new VALUE_TYPE[nrow];
    

#pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
    for(int i = 0; i < nrow; i++){
        double sumU = 0.0;
        for(int j = U_mtx->row_ptr[i]; j < U_mtx->row_ptr[i+1]; j++){
            sumU += U_mtx->values[j] * X[U_mtx->col_ind[j]];
        }
        LUvec[i] = sumU;
    }

    int k = 2;
    while(k < power_k){
        //L
        for(int c = 0; c < COLORS; c++){
#pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
            for(int blk = colors_ptr[c]; blk < colors_ptr[c + 1]; blk++){
                for(int i = blocks_ptr[blk]; i < blocks_ptr[blk + 1]; i++){
                    double sumL[2] = {0.0, 0.0};
                    for(int j = L_mtx->row_ptr[i]; j < L_mtx->row_ptr[i+1]; j++){
                        sumL[0] += L_mtx->values[j] * X[L_mtx->col_ind[j]];
                        sumL[1] += L_mtx->values[j] * Y[L_mtx->col_ind[j]];
                    }
                    //Y[k - 1][i] = (LUvec[i] + sumL[0] + dia_values[i] * Y[k - 2][i]) / info.max[k - 2];
                    Y[i] = (LUvec[i] + sumL[0] + dia_values[i] * X[i]);
                    LUvec[i] = sumL[1];
                }
            }
        }
        copy_vector(X, Y, nrow,nthreads);

        //U
        for(int c = COLORS - 1; c >= 0; c--){
 #pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
            for(int blk = colors_ptr[c + 1] - 1; blk >= colors_ptr[c]; blk--){
                for(int i = blocks_ptr[blk + 1] - 1; i >= blocks_ptr[blk]; i--){
                    double sumU[2] = {0.0, 0.0};
                    for(int j = U_mtx->row_ptr[i + 1] - 1; j >= U_mtx->row_ptr[i]; j--){
                        sumU[0] += U_mtx->values[j] * X[U_mtx->col_ind[j]];
                        sumU[1] += U_mtx->values[j] * Y[U_mtx->col_ind[j]]; 
                    }
                    //Y[k][i] = (LUvec[i] + sumU[0] + dia_values[i] * Y[k - 1][i]) / info.max[k - 1];
                    Y[i] = (LUvec[i] + sumU[0] + dia_values[i] * X[i]);
                    LUvec[i] = sumU[1];
                }
            }
        }
        copy_vector(X, Y, nrow,nthreads);
        k += 2;
    }

    //L
    for(int c = 0; c < COLORS; c++){
#pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
        for(int blk = colors_ptr[c]; blk < colors_ptr[c + 1]; blk++){
            for(int i = blocks_ptr[blk]; i < blocks_ptr[blk + 1]; i++){
                double sumL[2] = {0.0, 0.0};
                for(int j = L_mtx->row_ptr[i]; j < L_mtx->row_ptr[i + 1]; j++){
                    sumL[0] += L_mtx->values[j] * X[L_mtx->col_ind[j]];
                    sumL[1] += L_mtx->values[j] * Y[L_mtx->col_ind[j]];
                }
                //Y[power_k - 1][i] = (LUvec[i] + sumL[0] + dia_values[i] * Y[power_k - 2][i]) / info.max[power_k - 2];
                Y[i] = (LUvec[i] + sumL[0] + dia_values[i] * X[i]);
                LUvec[i] = sumL[1];
            }
        }
    }
    copy_vector(X, Y, nrow,nthreads);

    //最后一次迭代，补U
    #pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
    for(int i = 0; i < nrow; i++){
        double sumU = 0.0;
        for(int j = U_mtx->row_ptr[i]; j < U_mtx->row_ptr[i+1]; j++){
            sumU += U_mtx->values[j] * X[U_mtx->col_ind[j]];
        }
        //Y[power_k][i] = (sumU + LUvec[i] + dia_values[i] * Y[power_k - 1][i]) / info.max[power_k - 1];
        Y_fb[i] = (sumU + LUvec[i] + dia_values[i] * X[i]);
    }

    return;
}



int noBtB(csr_mtx* L_mtx, csr_mtx* U_mtx, VALUE_TYPE* dia_values, VALUE_TYPE *X, ABMC_info info, VALUE_TYPE *Y_fb)
{ 
    if(info.power_k % 2 == 1){
        cmp_fb_odd(L_mtx, U_mtx, dia_values, X, info, Y_fb);
    }
    else{
        cmp_fb_even(L_mtx, U_mtx, dia_values, X, info, Y_fb);
    }
    return 0;
}


#endif