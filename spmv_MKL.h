
#ifndef SPMV_MKL_H
#define SPMV_MKL_H

#include <mkl.h>
#include "mkl_spblas.h"
#include <omp.h>

double MPK_mkl(csr_mtx* mtx, ABMC_info info, double* X, double *Y)
{
    int nrow = mtx->nrow;
    int nnz = mtx->nnz;
    int nthreads = info.nthreads;
    int power_k = info.power_k;

    double *values;
    int* row_ptr_b, * row_ptr_e, * col_ind;
    double* y, * x;
    x = X;

    sparse_status_t stat;

    matrix_descr tt;
    tt.type = SPARSE_MATRIX_TYPE_GENERAL;
    //tt.mode = SPARSE_FILL_MODE_LOWER;
    tt.diag = SPARSE_DIAG_NON_UNIT;

    double alpha = 1.0;
    double beta = 0.0;
    values = (double*)mkl_malloc(nnz * sizeof(double), 64);
    col_ind = (int*)mkl_malloc(nnz * sizeof(int), 64);
    row_ptr_b = (int*)mkl_malloc(nrow * sizeof(int), 64);
    row_ptr_e = (int*)mkl_malloc(nrow * sizeof(int), 64);
    y = (double*)mkl_malloc(nrow * sizeof(double), 64);
    x = (double*)mkl_malloc(nrow * sizeof(double), 64);

    for(int i = 0; i < nnz; i++){
        values[i] = mtx->values[i];
        col_ind[i] = mtx->col_ind[i];
    }
    for(int i = 0; i < nrow; i++){
        row_ptr_b[i] = mtx->row_ptr[i];
        row_ptr_e[i] = mtx->row_ptr[i + 1];
    }

    sparse_matrix_t A;
    mkl_set_num_threads(nthreads);
    stat = mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, nrow, nrow, row_ptr_b, row_ptr_e, col_ind, values);
    if(stat != SPARSE_STATUS_SUCCESS){
        cout << "create failed" << endl;
    }
	

    double start = mytimer();

    for(int i = 0; i < power_k; i++){   
        stat = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, tt, x, beta, y);

#pragma omp parallel for num_threads(nthreads) schedule(guided) proc_bind(spread)
        for(int k = 0; k < nrow; k++){
            x[k] = y[k];
        }
    }
    
    double end = mytimer();
    
    for(int i = 0; i < nrow; i++){
        Y[i] = y[i];
    }
    return end - start;

}

#endif
