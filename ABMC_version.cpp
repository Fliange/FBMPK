
#include "ABMCpre.h"
#define NTIMES 30


#ifdef MKL
#include "spmv_MKL.h"
#endif

#ifdef BTB
#include "ABMC_noBtB_MPK.h"
#endif

#ifdef RESULTCHECK
#include "ResultCheck.h"
#endif


int main(int argc, char* argv[])
{
    char* file_in = 0;
    int power_k;
    int nthreads;
    int part_num = 1024;

    if(argv[1]==NULL){
        printf("\tUsage: ./abmc_MPK xxx.mtx power_k nthreads part_num(1024)\n");
        return -1;
    }
    printf("\t***ABMC****\n");
    file_in = argv[1];
    if(argv[2] == NULL){
        printf("Usage: please input power_k\n");
        return -1;
    }
    power_k = strtol(argv[2], NULL, 10);
    printf("\tNumber of power K: %i\n", power_k);
    if(argv[3] == NULL){
        printf("Usage: please input number of threads\n");
        return -1;
    }
    nthreads = strtol(argv[3], NULL, 10);
    printf("\tNumber of Threads:%i\n", nthreads);
    //if(argv[4] != NULL){
    //    part_num = strtol(argv[4], NULL, 10);
    //}
    //printf("\tNumber of part_num:%i\n", part_num);
    printf("\n");


    csr_mtx mtx;
    readmtx(file_in, &mtx);
    ABMC_info info;
    info.power_k = power_k;
    info.nthreads = nthreads;

    
    csr_mtx abmc_mtx;
    //ABMCpre(&mtx, &abmc_mtx, &info, part_num);
    readABMC(&abmc_mtx, file_in, &info);
    

    csr_mtx L_abmc_mtx, U_abmc_mtx;
    VALUE_TYPE* dia_values = new VALUE_TYPE[mtx.nrow];
    for(int i = 0; i < mtx.nrow; i++){
        dia_values[i] = 0.0;
    }
    split_LDU(&abmc_mtx, &L_abmc_mtx, &U_abmc_mtx, dia_values);
    

    VALUE_TYPE* X = new VALUE_TYPE[mtx.nrow];

    
    double baseline_time = 0.0;
    VALUE_TYPE* Y_baseline = new VALUE_TYPE[mtx.nrow];
    for(int n = 0; n < NTIMES; n++){
#pragma omp parallel for num_threads(nthreads) proc_bind(spread)
        for(int i = 0; i < mtx.nrow; i++){
            X[i] = 0.0001;
            Y_baseline[i] = 0.0;
        }
        double bs_start = mytimer();
        ABMC_baseline(&mtx, info, X, Y_baseline);
        double bs_end = mytimer();
        baseline_time += bs_end - bs_start;
    }
    printf("\tBaseline finished..\n");



    double fbABMC_time = 0.0;
    VALUE_TYPE* Y_fb = new VALUE_TYPE[mtx.nrow];
    for(int n = 0; n < NTIMES; n++){
#pragma omp parallel for num_threads(nthreads) schedule(guided) proc_bind(spread)
        for(int i = 0; i < mtx.nrow; i++){
            X[i] = 0.0001;
            Y_fb[i] = 0.0;
        }
    VALUE_TYPE* xy = new VALUE_TYPE[mtx.nrow * 2];
#pragma omp parallel for num_threads(nthreads) 
    for(int i = 0; i < mtx.nrow; i++){
        xy[i * 2] = X[i];//x初始化在偶数位
    }
        double fb_start = mytimer();
        ABMC_fb(&L_abmc_mtx, &U_abmc_mtx, dia_values, xy, info, Y_fb);
        double fb_end = mytimer();
        fbABMC_time += fb_end - fb_start;
    }
    printf("\tabmc forwardbackward finished..\n");


#ifdef BTB
    double noBtB_time = 0.0;
    VALUE_TYPE* Y_noBtB = new VALUE_TYPE[mtx.nrow];
    for(int n = 0; n < NTIMES; n++){
#pragma omp parallel for num_threads(nthreads) schedule(guided) proc_bind(spread)
        for(int i = 0; i < mtx.nrow; i++){
            X[i] = 0.0001;
            Y_noBtB[i] = 0.0;
        }
        double noBtB_start = mytimer();
        noBtB(&L_abmc_mtx, &U_abmc_mtx, dia_values, X, info, Y_noBtB);
        double noBtB_end = mytimer();
        noBtB_time += noBtB_end - noBtB_start;
    }
    printf("\tnoBtB finished..\n");
#endif

    
#ifdef MKL
    double mkl_time = 0.0;
    VALUE_TYPE* Y_mkl = new VALUE_TYPE[mtx.nrow];   
    for(int n = 0; n < NTIMES; n++){
#pragma omp parallel for num_threads(nthreads) schedule(guided) proc_bind(spread)
        for(int i = 0; i < mtx.nrow; i++){
            X[i] = 0.0001;
            Y_mkl[i] = 0.0;
        }
        mkl_time += MPK_mkl(&mtx, X, info, Y_mkl);
}
#endif


    cout << "baseline:" << endl;
#ifdef MKL
    cout << "mkl:" << endl;
#endif
#ifdef BTB
    cout << "fb: " << endl;
#endif
    cout << "fb+BtB:" << endl;

    
    cout << baseline_time / NTIMES << endl;
#ifdef MKL
    cout << mkl_time / NTIMES << endl;
#endif
#ifdef BTB
    cout << noBtB_time / NTIMES << endl;
#endif 
    cout << fbABMC_time / NTIMES << endl;

#ifdef RESULTCHECK
    cout << "Y_fb" << endl;
    check(Y_baseline, Y_fb, mtx.nrow, info);
    cout << "Y_noBtB" << endl;
    check(Y_baseline, Y_noBtB, mtx.nrow, info);
    cout << "Y_abmc_base" << endl;
    VALUE_TYPE* Y_abmc_base = new VALUE_TYPE[mtx.nrow];
    for(int i = 0; i < mtx.nrow; i++){
            X[i] = 0.0001;
            Y_abmc_base[i] = 0.0;
        }
    ABMC_baseline(&abmc_mtx, info, X, Y_abmc_base);
    check(Y_baseline, Y_abmc_base, mtx.nrow, info);
#endif
}