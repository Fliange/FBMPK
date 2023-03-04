# FBMPK

Contact: Yichen Zhang(zhangyichen@nudt.edu.cn)

FBMPK presents a novel approach to optimize multiple invocations of a sparse matrix-vector multiplication(SpMV) kernel performed on the same sparse matrix A and dense
vector x, like $Ax$, $A^2x$, · · · , $A^kx$, and their linear combinations such as $Ax + A^2x$. It achieve this by partitioning the sparse matrix into submatrices and devising a new computation pipeline that reduces memory access to the sparse matrix and exploits the data locality of the dense vector of SpMV. 

# Paper information

Yichen Zhang, Shengguo Li, Fan Yuan, Dezun Dong, Xiaojian Yang, Tiejun Li, Zheng Wang: Memory-aware Optimization for Sequences of Sparse Matrix-Vector Multiplications

# Software dependences
GNU Compiler(GCC) V8.2
Intel oneAPI 2022.1.2
OpenMP

#Getting Started
The makefile corresponding to this program:

    #CXX=icc
    CXX=g++
    #CXXFLAGS+=-O3 -ffast-math -qopenmp -std=c++11
    CXXFLAGS+=-O3 -ffast-math -fopenmp -std=c++11

    #CXXFLAGS+= -DRESULTCHECK
    CXXFLAGS+= -DBTB #-DKAHAN

    #MKLFLAGS+= -I/home/zyc/intel/oneapi/mkl/2022.0.2/include -L/home/zyc/intel/oneapi/mkl/2022.0.2/lib/intel64
    #MKLFLAGS+= -I/usr/local/bin/../include/ -L/usr/local/bin/../bin/ #-DLIKWID_PERFMON
    #MKLFLAGS+= -lmkl_core -lmkl_intel_thread -liomp5 -lmkl_intel_lp64
    #MKLFLAGS+= -MKL

    all: abmc_MPK 

    %.o: %.cpp
	    $(CXX) $(INCLUDES) $(CXXFLAGS) $(MKLFLAGS) -c $< -o $@

    abmc_MPK: $(OBJ) 
	     $(CXX) ABMC_version.cpp $^ $(INCLUDES) $(CXXFLAGS) -o $@
     # $(CXX) ABMC_version.cpp $^ $(INCLUDES) $(CXXFLAGS) $(MKLFLAGS) ABMC_version.cpp -o $@


    clean:
	    rm -f $(OBJ) abmc_MPK
    .PHONY: abmc_MPK 
    
 Note that in order to use mkl on your machine, you need to modify MKLFLAGS.
 This part of the code does not include preprocessing. audikw_1.zip is the matrix we reordered with ABMC in advance, you can use this matrix for verification.
 
