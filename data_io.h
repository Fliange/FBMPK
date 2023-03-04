
#ifndef DATA_IO_H
#define DATA_IO_H

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <sys/resource.h>
#include <omp.h>

#include "mmio.h"

#define VALUE_TYPE double
using namespace std;


typedef struct csr_mtx//由于重排后上三角出现了元素，这里没有严格按照从左往右的顺序存储
{
    int nrow;
    int nnz;
    int* row_ptr;
    int *col_ind;
    VALUE_TYPE *values;
}csr_mtx;

void copy_vector(VALUE_TYPE* X, VALUE_TYPE* Y, int size, VALUE_TYPE max, int nthreads)
{
#pragma omp parallel for num_threads(nthreads) //schedule(guided) proc_bind(spread)
    for(int i = 0; i < size; i++){
        Y[i] = Y[i] / max;
        X[i] = Y[i];
    }
}



double mytimer(void)
{
	struct timeval tp;
	static long start=0, startu;
	if (!start) {
		gettimeofday(&tp, NULL);
		start = tp.tv_sec;
		startu = tp.tv_usec;
		return 0.0;
	}
	gettimeofday(&tp, NULL);
	return ((double) (tp.tv_sec - start)) + (tp.tv_usec-startu)/1000000.0 ;
}


int read_info(int* nrow, int* nnz, int *isSymmetric, const char* filename)
{
    int ret_code;
    MM_typecode matcode;
    int M, N, nz;   
    FILE* fp;

	//printf("\tOpening matrix market file\n");
	if ((fp = fopen(filename, "r")) == NULL){
		printf("***Failed to open MatrixMarket file %s ***\n", filename);
        exit(1);
	}
	
	//printf("\tReading MatrixMarket banner\n");
	if (mm_read_banner(fp, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner\n");
        exit(1);
    }

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }
  	
	//printf("\tReading sparse matrix size...\n");
    if ((ret_code = mm_read_mtx_crd_size(fp, &N, &M, &nz)) !=0)
        exit(1);
    if(M != N){
        printf("Sorry, matrix is not square\n ");
        exit(1);
    }
    
    *nrow = M;
    *nnz = nz;
    *isSymmetric = 0;
    if(matcode[3] == 'S')
        *isSymmetric = 1;
    
    return 0;
}


int read_data(int* row_ind, int* col_ind, VALUE_TYPE* values, int *dia, int *L, int *R, const char* filename)
{
    int ret_code;
    int M, N, nz;   
    FILE* fp;

	if ((fp = fopen(filename, "r")) == NULL){
		printf("***Failed to open MatrixMarket file %s ***\n", filename);
        exit(1);
    }
    if ((ret_code = mm_read_mtx_crd_size(fp, &N, &M, &nz)) !=0)
        exit(1);
    
    int di = 0, Ln = 0, Rn = 0;
    int fsc;
    //printf("\tReading matrix entries from file\n");
    for(int i = 0; i < nz; i++){
        fsc = fscanf(fp, "%d %d %lg\n", &row_ind[i], &col_ind[i], &values[i]);
		row_ind[i]--;
        col_ind[i]--;
        if(row_ind[i] == col_ind[i]){
            di++;
        }
        if(row_ind[i] > col_ind[i]){
            Ln++;
        }
        if(row_ind[i] < col_ind[i]){
            Rn++;
        }
    }
    if(fp != stdin) fclose(fp);

    *L = Ln;
    *R = Rn;
    *dia = di;

    return 0;
}


void vec_write(char* filename, VALUE_TYPE* Y, int nrow)
{
	FILE* fp = fopen(filename, "w");

	fprintf(fp, "%d", nrow);

	for (int i = 0; i < nrow; ++i)
		fprintf(fp, "\n%20.16g", Y[i]);

	fclose(fp);
}



int readmtx(const char* filename, csr_mtx* sourcematrix)
{
    printf("\tReading Matrix\n");

    int nrow, nnz, dia, Lnnz, Rnnz;
    int* row_ind, * col_ind;
    VALUE_TYPE* values;
    int isSymmetric;

    read_info(&nrow, &nnz, &isSymmetric, filename);
    row_ind = new int[nnz];
    col_ind = new int[nnz];
    values = new VALUE_TYPE[nnz];
    read_data(row_ind, col_ind, values, &dia, &Lnnz, &Rnnz, filename);

    /*Symmetric*/
    int* row_ind_tmp, * col_ind_tmp;
    VALUE_TYPE* values_tmp;
    if(isSymmetric){
        row_ind_tmp = new int[nnz * 2 - dia];
        col_ind_tmp = new int[nnz * 2 - dia];
        values_tmp = new VALUE_TYPE[nnz * 2 - dia];

        int j = 0;
        for(int k = 0; k < nnz; ++k){
            row_ind_tmp[k] = row_ind[k];
            col_ind_tmp[k] = col_ind[k];
            values_tmp[k] = values[k];
            if(row_ind[k] != col_ind[k]){
                row_ind_tmp[nnz + j] = col_ind[k];
                col_ind_tmp[nnz + j] = row_ind[k];
                values_tmp[nnz + j] = values[k];
                j++;
            }
        }
        Lnnz = nnz - dia;
        Rnnz = nnz - dia;
        nnz = nnz * 2 - dia;
        delete[] row_ind, col_ind, values;
    }
    else{
        row_ind_tmp = row_ind;
        col_ind_tmp = col_ind;
        values_tmp = values;
    }

    //convert into csr
	int* Bp = new int[nrow + 1];
	int* Bi = new int[nnz];
	VALUE_TYPE* Bv = new VALUE_TYPE[nnz];
	for (int i = 0; i <= nrow; ++i)
        Bp[i] = 0;
	for (int j = 0; j < nnz; ++j)
        ++Bp[row_ind_tmp[j]];
	for (int i = 0; i < nrow; ++i)
        Bp[i + 1] += Bp[i];
	for (int j = nnz - 1; j >= 0; --j){
		Bi[--Bp[row_ind_tmp[j]]] = col_ind_tmp[j];
		Bv[Bp[row_ind_tmp[j]]] = values_tmp[j];
    }

    sourcematrix->nnz = nnz;
    sourcematrix->nrow = nrow;
    sourcematrix->row_ptr = Bp;
    sourcematrix->col_ind = Bi;
    sourcematrix->values = Bv;

    delete[] row_ind_tmp, col_ind_tmp, values_tmp;
    return 0;
}



int split_LDU(csr_mtx* sourcemtx, csr_mtx* L_mtx, csr_mtx* U_mtx, VALUE_TYPE* dia_values)
{
    printf("\tMatrix Spliting\n");
    int nrow = sourcemtx->nrow;
	int* Ap = sourcemtx->row_ptr;
	int* Ai = sourcemtx->col_ind;
    VALUE_TYPE* Av = sourcemtx->values;

    VALUE_TYPE* Dv = dia_values;

    int* Lp = new int[nrow + 1];
    int* Up = new int[nrow + 1];
	Lp[0] = 0;
    Up[0] = 0;
    
#pragma omp parallel for schedule(guided)
	for (int i = 0; i < nrow; ++i){
		int cntl = 0;
		int cntu = 0;
		for (int j = Ap[i]; j < Ap[i + 1]; ++j){
			int jcol = Ai[j];
			if (jcol < i) 
				++cntl;
			else if (jcol > i)
				++cntu;
		}
		Lp[i + 1] = cntl;
		Up[i + 1] = cntu;
    }
    for(int i = 0; i < nrow; ++i){
        Lp[i + 1] += Lp[i];
    }
    for(int i = 0; i < nrow; ++i){
        Up[i + 1] += Up[i];
    }

    int* Li = new int[Lp[nrow]];
	double* Lv = new double[Lp[nrow]];
	int* Ui = new int[Up[nrow]];
    double* Uv = new double[Up[nrow]];

#pragma omp parallel for schedule(guided)
	for (int i = 0; i < nrow; ++i){
		for (int j = Ap[i], k = Lp[i], r = Up[i]; j < Ap[i + 1]; ++j){
			int jcol = Ai[j];
			double jval = Av[j];
			if (jcol < i) {
				Li[k] = jcol;
				Lv[k++] = jval;
			}
			else if (jcol > i){
				Ui[r] = jcol;
				Uv[r++] = jval;
			}
            else if(jcol == i){
                Dv[i] = jval;
            }
        }
    }

    L_mtx->nrow = nrow;
    L_mtx->row_ptr = Lp;
    L_mtx->col_ind = Li;
    L_mtx->values = Lv;
    L_mtx->nnz = L_mtx->row_ptr[nrow];

    U_mtx->nrow = nrow;
    U_mtx->row_ptr = Up;
    U_mtx->col_ind = Ui;
    U_mtx->values = Uv;
    U_mtx->nnz = U_mtx->row_ptr[nrow];

    return 0;
}

#endif