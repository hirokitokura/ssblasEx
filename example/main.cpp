#include<iostream>
#include<random>
#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
#include "fj_tool/fapp.h" 
#endif


#include "../include/ssblasBatch.h"
#include"./include/getters.hpp"
#include"./include/init_mat_batch.hpp"
#include"./include/check_ans_batch.hpp"
#include"./include/naive_gemm.hpp"


void print_perf(
    const int M,
    const int N,
    const int K,
    const int BATCH,
    double time_s
)
{
    double Gflops=0;
    Gflops=2.0*(double)M*(double)N*(double)K*(double)BATCH;
    Gflops/=1000*1000*1000;
    Gflops/=time_s;
    printf("%d,%d,%d,%f,[s],%f,[Gflops]\n", M,N,K,time_s,Gflops);
}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void test_func
(
    const ssblasOperation_t transA,
    const ssblasOperation_t transB,
    const T_SCALE alpha,
    const INDEXINT M,
    const INDEXINT N,
    const INDEXINT K,
    const T_SCALE beta,
    const INDEXINT lda,
    const INDEXINT ldb,
    const INDEXINT ldc,
    const INDEXINT BATCH
)
{

    DEF_DOUBLE_AB** A;
    DEF_DOUBLE_AB** B;
    DEF_DOUBLE_C** C_NAIVE;
    DEF_DOUBLE_C** C_TUNE;

    A=(DEF_DOUBLE_AB**)malloc(sizeof(DEF_DOUBLE_AB*)*BATCH);
    B=(DEF_DOUBLE_AB**)malloc(sizeof(DEF_DOUBLE_AB*)*BATCH);
    C_NAIVE=(DEF_DOUBLE_C**)malloc(sizeof(DEF_DOUBLE_C*)*BATCH);
    C_TUNE=(DEF_DOUBLE_C**)malloc(sizeof(DEF_DOUBLE_C*)*BATCH);
    for(INDEXINT i=0;i<BATCH;i++){
        A[i]=(DEF_DOUBLE_AB*)malloc(sizeof(DEF_DOUBLE_AB)*lda*(SSBLAS_OP_N==transA?K:M));
        B[i]=(DEF_DOUBLE_AB*)malloc(sizeof(DEF_DOUBLE_AB)*ldb*(SSBLAS_OP_N==transB?N:K));
        C_NAIVE[i]=(DEF_DOUBLE_C*)malloc(sizeof(DEF_DOUBLE_C)*ldc*N);
        C_TUNE[i]=(DEF_DOUBLE_C*)malloc(sizeof(DEF_DOUBLE_C)*ldc*N);
    }

    if(SSBLAS_OP_N==transA){
        init_mat_batch<INDEXINT>(M,K,A,lda,BATCH);
    }
    else if(SSBLAS_OP_T==transA){
        init_mat_batch<INDEXINT>(K,M,A,lda,BATCH);
    }

    if(SSBLAS_OP_N==transB){
        init_mat_batch<INDEXINT>(K,N,B,ldb, BATCH);
    }
    else if(SSBLAS_OP_T==transB){
        init_mat_batch<INDEXINT>(N,K,B,ldb, BATCH);
    }

    

    init_mat_batch<INDEXINT>(M,N,C_NAIVE,ldc, BATCH);
    for(INDEXINT i=0;i<BATCH;i++){
        //memset(C[0][i], 1, sizeof(float)*ldc*n);
        memcpy(C_TUNE[i], C_NAIVE[i], sizeof(DEF_DOUBLE_C)*ldc*N);
    }

    double start, end;
    double time_s[2];
    printf("naive_gemm start\n");
#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
fapp_start("naive_gemm",1,0);
#endif
    start=getTimeinS();
    naive_gemm
    <INDEXINT, T_SCALE,DEF_DOUBLE_AB,DEF_DOUBLE_C>
    (
        transA,
        transB,
        M,
        N,
        K,
        alpha,
        A,
        lda,
        B,
        ldb,
        beta,
        C_NAIVE,
        ldc,
        BATCH
    );
    end=getTimeinS();
#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
fapp_stop("naive_gemm",1,0);
#endif
    time_s[0]=(end-start);
    printf("naive_gemm end\n");

    printf("ssblasGemmEx start\n");
    ssblasStatus_t retval;
#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
fapp_start("ssblasGemmBatchedEx",1,0);
#endif
    start=getTimeinS();
    if(1)
    retval=ssblasGemmBatchedEx(
        transA,
        transB,
        M,
        N,
        K,
        &alpha,
        (const void* const*)A,
        //SSBLAS_R_32F,
        getDataType<DEF_DOUBLE_AB>(),
        lda,
         (const void* const*)B,
        //SSBLAS_R_32F,
        getDataType<DEF_DOUBLE_AB>(),
        ldb,
        &beta,
        (void* const*)C_TUNE,
        //SSBLAS_R_32F,
        getDataType<DEF_DOUBLE_C>(),
        ldc,
        BATCH,
        SSBLAS_COMPUTE_DEFAULT_TYPE,
        SSBLAS_COMPUTE_DEFAULT
    );
    end=getTimeinS();
    time_s[1]=(end-start);
#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
fapp_stop("ssblasGemmBatchedEx",1,0);
#endif
    printf("retval: %d\n",0);
    ssblasShowError(retval);
    check_ans_batch<INDEXINT>(
        M,N,
        (const DEF_DOUBLE_C**)C_NAIVE,
        (const DEF_DOUBLE_C**)C_TUNE,
        ldc,
        BATCH
    );

    print_perf(M,N,K,BATCH, time_s[0]);
    print_perf(M,N,K,BATCH, time_s[1]);
    
    for(int i=0;i<BATCH;i++){
        free(A[i]);
        free(B[i]);
        free(C_NAIVE[i]);
        free(C_TUNE[i]);
    }
    free(A);
    free(B);
    free(C_NAIVE);
    free(C_TUNE);
}

//Xgemm_kernel_sve_v4x5_1<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>




int main(int argc, char* argv[])
{
    //./a.out INT_TYPE SCALE_TYPE AB_TYPE C_TYPE TRANSA TRANSB BATCH alpha M N K beta
    //./a.out INT_TYPE SCALE_TYPE AB_TYPE C_TYPE TRANSA TRANSB BATCH alpha M N K beta lda ldb ldc
    ssblasIntType_t INT_type;
    ssblasDataType_t SCALE_type;
    ssblasDataType_t AB_type;
    ssblasDataType_t C_type;
    ssblasOperation_t transA;
    ssblasOperation_t transB;

    size_t BATCH;
    char* alpha_ptr=NULL;
    size_t M;
    size_t N;
    size_t K;
    char* beta_ptr=NULL;
    size_t lda;
    size_t ldb;
    size_t ldc;
    if(argc==13){
        //./a.out INT_TYPE SCALE_TYPE AB_TYPE C_TYPE TRANSA TRANSB BATCH alpha M N K beta
        INT_type=getINTTYPE(std::string(argv[1]));
        SCALE_type=getSCALETYPE(std::string(argv[2]));
        AB_type=getABTYPE(std::string(argv[3]));
        C_type=getCTYPE(std::string(argv[4]));
        transA=getTRANS(std::string(argv[5]));
        transB=getTRANS(std::string(argv[6]));

        BATCH=atoll(argv[7]);
        alpha_ptr=argv[8];
        M=atoll(argv[9]);
        N=atoll(argv[10]);
        K=atoll(argv[11]);
        beta_ptr=argv[12];
        if((SSBLAS_OP_N==transA)){
            lda=M;
        }
        else if((SSBLAS_OP_T==transA)){
            lda=K;
        }
        if((SSBLAS_OP_N==transB)){
            ldb=K;
        }
        else if((SSBLAS_OP_T==transB)){
            ldb=N;
        }
        
        ldc=M;
    }
    else if(argc==16){
        //./a.out INT_TYPE SCALE_TYPE AB_TYPE C_TYPE TRANSA TRANSB BATCH alpha M N K beta lda ldb ldc
        INT_type=getINTTYPE(std::string(argv[1]));
        SCALE_type=getSCALETYPE(std::string(argv[2]));
        AB_type=getABTYPE(std::string(argv[3]));
        C_type=getCTYPE(std::string(argv[4]));
        transA=getTRANS(std::string(argv[5]));
        transB=getTRANS(std::string(argv[6]));

        BATCH=atoll(argv[7]);
        alpha_ptr=argv[8];
        M=atoll(argv[9]);
        N=atoll(argv[10]);
        K=atoll(argv[11]);
        beta_ptr=argv[12];
        lda=atoll(argv[13]);
        ldb=atoll(argv[14]);
        ldc=atoll(argv[15]);
    }
    else{
        printf("./a.out INT_TYPE SCALE_TYPE AB_TYPE C_TYPE TRANSA TRANSB BATCH alpha M N K beta\n");
        printf("./a.out INT_TYPE SCALE_TYPE AB_TYPE C_TYPE TRANSA TRANSB BATCH alpha M N K beta lda ldb ldc\n");

        printf("INT_TYPE: int longint\n");
        printf("SCALE_TYPE: float int\n");
        printf("AB_TYPE: float char\n");
        printf("C_TYPE: float int\n");
        exit(-1);
    }


    if(SCALE_type==SSBLAS_R_32F && AB_type==SSBLAS_R_32F && C_type==SSBLAS_R_32F){
        if(INT_type==SSBLAS_int_32){
            //printf("INT_type: SSBLAS_int_32\n");
        }
        else if(INT_type==SSBLAS_int_64){
            //printf("INT_type: SSBLAS_int_64\n");
        }
        else{
            printf("UNKNOWN INT TYPE\n");
            exit(-1);
        }
        /*printf("SCALE_type: SSBLAS_R_32F\n");
        printf("AB_type: SSBLAS_R_32F\n");
        printf("C_type: SSBLAS_R_32F\n");
        printf("alpha: %f\n",atof(alpha_ptr));
        printf("beta: %f\n",atof(beta_ptr));
        printf("M: %d\n", (int)M);
        printf("N: %d\n", (int)N);
        printf("K: %d\n", (int)K);
        printf("lda: %d\n", (int)lda);
        printf("ldb: %d\n", (int)ldb);
        printf("ldc: %d\n", (int)ldc);*/
        if(INT_type==SSBLAS_int_32){
            test_func<SSINT, float, float, float>
            (
                transA,
                transB,
                atof(alpha_ptr),
                M,
                N,
                K,
                atof(beta_ptr),
                lda,
                ldb,
                ldc,
                BATCH
            );
        }
        else if(INT_type==SSBLAS_int_64){
            test_func<SSLONG, float, float, float>
            (
                transA,
                transB,
                atof(alpha_ptr),
                M,
                N,
                K,
                atof(beta_ptr),
                lda,
                ldb,
                ldc,
                BATCH
            );
        }
            
        
    }
    else if(SCALE_type==SSBLAS_R_64F && AB_type==SSBLAS_R_64F && C_type==SSBLAS_R_64F){
        if(INT_type==SSBLAS_int_32){
            //printf("INT_type: SSBLAS_int_32\n");
        }
        else if(INT_type==SSBLAS_int_64){
            //printf("INT_type: SSBLAS_int_64\n");
        }
        else{
            printf("UNKNOWN INT TYPE\n");
            exit(-1);
        }
        /*printf("SCALE_type: SSBLAS_R_64F\n");
        printf("AB_type: SSBLAS_R_64F\n");
        printf("C_type: SSBLAS_R_64F\n");
        printf("alpha: %f\n",atof(alpha_ptr));
        printf("beta: %f\n",atof(beta_ptr));
        printf("M: %d\n", (int)M);
        printf("N: %d\n", (int)N);
        printf("K: %d\n", (int)K);
        printf("lda: %d\n", (int)lda);
        printf("ldb: %d\n", (int)ldb);
        printf("ldc: %d\n", (int)ldc);*/
        if(INT_type==SSBLAS_int_32){
            test_func<SSINT, double, double, double>
            (
                transA,
                transB,
                atof(alpha_ptr),
                M,
                N,
                K,
                atof(beta_ptr),
                lda,
                ldb,
                ldc,
                BATCH
            );
        }
        else if(INT_type==SSBLAS_int_64){
            test_func<SSLONG, double, double, double>
            (
                transA,
                transB,
                atof(alpha_ptr),
                M,
                N,
                K,
                atof(beta_ptr),
                lda,
                ldb,
                ldc,
                BATCH
            );
        }
            
        
    }
    else if(SCALE_type==SSBLAS_R_32I && AB_type==SSBLAS_R_8I && C_type==SSBLAS_R_32I){
        //printf("SDOT\n");
        if(INT_type==SSBLAS_int_32){
            //printf("INT_type: SSBLAS_int_32\n");
        }
        else if(INT_type==SSBLAS_int_64){
            //printf("INT_type: SSBLAS_int_64\n");
        }
        else{
            printf("UNKNOWN INT TYPE\n");
            exit(-1);
        }
        /*printf("SCALE_type: SSBLAS_R_32I\n");
        printf("AB_type: SSBLAS_R_8I\n");
        printf("C_type: SSBLAS_R_32I\n");
        printf("alpha: %d\n",atoi(alpha_ptr));
        printf("beta: %d\n",atoi(beta_ptr));
        printf("M: %d\n", (int)M);
        printf("N: %d\n", (int)N);
        printf("K: %d\n", (int)K);
        printf("lda: %d\n", (int)lda);
        printf("ldb: %d\n", (int)ldb);
        printf("ldc: %d\n", (int)ldc);*/
        if(INT_type==SSBLAS_int_32){
            test_func<SSINT, int, char, int>
            (
                transA,
                transB,
                atoi(alpha_ptr),
                M,
                N,
                K,
                atoi(beta_ptr),
                lda,
                ldb,
                ldc,
                BATCH
            );
        }
        else if(INT_type==SSBLAS_int_64){
            test_func<SSLONG, int, char, int>
            (
                transA,
                transB,
                atoi(alpha_ptr),
                M,
                N,
                K,
                atoi(beta_ptr),
                lda,
                ldb,
                ldc,
                BATCH
            );
        }
    }
    else{
        printf("UNSUPPORT TYPE\n");
        exit(-1);
    }


    return 0;
}