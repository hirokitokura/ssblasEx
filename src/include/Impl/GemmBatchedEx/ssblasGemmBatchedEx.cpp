#include<iostream>

#include "../../../../include/ssblasBatch.h"
#include "ssblasGemmBatchedExSelector.hpp"



void print_ssblasOperation_t(ssblasOperation_t transa)
{
    if(SSBLAS_OP_N==transa){
        printf("SSBLAS_OP_N");
    }
    else if(SSBLAS_OP_T==transa){
        printf("SSBLAS_OP_T");
    }
    else if(SSBLAS_OP_C==transa){
        printf("SSBLAS_OP_C");
    }
    else{
        printf("UNKNOWN");
    }
}

void print_ssblasDataType_t(ssblasDataType_t Atype)
{
    if(SSBLAS_R_8I==Atype)printf("SSBLAS_R_8I");
    else if(SSBLAS_R_8U==Atype)printf("SSBLAS_R_8U");
    else if(SSBLAS_R_16F==Atype)printf("SSBLAS_R_16F");
    else if(SSBLAS_R_32I==Atype)printf("SSBLAS_R_32I");
    else if(SSBLAS_R_32F==Atype)printf("SSBLAS_R_32F");
    else if(SSBLAS_R_64F==Atype)printf("SSBLAS_R_64F");
    else printf("UNKNONW");
}

void print_ssblasComputeType_t(ssblasComputeType_t computeType)
{
    if(SSBLAS_COMPUTE_DEFAULT_TYPE==computeType)printf("SSBLAS_COMPUTE_DEFAULT_TYPE");
    else if(SSBLAS_COMPUTE_32I==computeType)printf("SSBLAS_COMPUTE_32I");
    else if(SSBLAS_COMPUTE_32F==computeType)printf("SSBLAS_COMPUTE_32F");
    else printf("UNKNONW");
}
void print_ssblasGemmAlgo_t(ssblasGemmAlgo_t algo)
{
    if(SSBLAS_COMPUTE_DEFAULT==algo)printf("SSBLAS_COMPUTE_DEFAULT");
    else printf("UNKNONW");
}

void ssblasGemmBatchedExShowdebug
(
    ssblasOperation_t    transa,
    ssblasOperation_t    transb,
    SSINT                  m,
    SSINT                  n,
    SSINT                  k,
    const void*          alpha,
    const void *const    A[],
    ssblasDataType_t     Atype,
    const SSINT            lda,
    const void *const    B[],
    ssblasDataType_t     Btype,
    const SSINT            ldb,
    const void*          beta,
    void *const          C[],
    ssblasDataType_t     Ctype,
    const SSINT            ldc,
    const SSINT            batchCount,
    ssblasComputeType_t  computeType,
    ssblasGemmAlgo_t     algo
)
{
    printf("ssblasGemmBatchedExShowdebug: START\n");
    printf("transa: ");print_ssblasOperation_t(transa);printf("\n");
    printf("transb: ");print_ssblasOperation_t(transb);printf("\n");
    std::cout << "M: " << m << std::endl; 
    std::cout << "N: " << n << std::endl; 
    std::cout << "K: " << k << std::endl; 
    std::cout << "alpha: " << "" << std::endl; 
    std::cout << "A: " << "" << std::endl; 
    printf("Atype: ");print_ssblasDataType_t(Atype);printf("\n");
    std::cout << "lda: " << lda << std::endl; 
    std::cout << "B: " << "" << std::endl; 
    printf("Btype: ");print_ssblasDataType_t(Btype);printf("\n");
    std::cout << "ldb: " << ldb << std::endl; 
    std::cout << "beta: " << "" << std::endl; 
    std::cout << "C: " << "" << std::endl; 
    printf("Ctype: ");print_ssblasDataType_t(Ctype);printf("\n");
    std::cout << "ldc: " << ldc << std::endl; 
    std::cout << "batchCount: " << batchCount << std::endl; 

    printf("computeType: ");print_ssblasComputeType_t(computeType);printf("\n");
    printf("algo: ");print_ssblasGemmAlgo_t(algo);printf("\n");
    printf("ssblasGemmBatchedExShowdebug: END\n");
}



ssblasStatus_t ssblasGemmEx_main(
    ssblasOperation_t    transa,
    ssblasOperation_t    transb,
    SSINT                  m,
    SSINT                  n,
    SSINT                  k,
    const void*          alpha,
    const void *const    A[],
    ssblasDataType_t     Atype,
    const SSINT            lda,
    const void *const    B[],
    ssblasDataType_t     Btype,
    const SSINT            ldb,
    const void*          beta,
    void *const          C[],
    ssblasDataType_t     Ctype,
    const SSINT            ldc,
    const SSINT            batchCount,
    ssblasComputeType_t  computeType,
    ssblasGemmAlgo_t     algo
)
{
    ssblasStatus_t retval;
    //showdebug
    //SSBLAS_GEMMBATCHEDEX_DEBUG
    {
        char* SSBLAS_GEMMBATCHEDEX_DEBUG=getenv("SSBLAS_GEMMBATCHEDEX_DEBUG");
        if(NULL!=SSBLAS_GEMMBATCHEDEX_DEBUG){
            ssblasGemmBatchedExShowdebug
            (
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                A,
                Atype,
                lda,
                B,
                Btype,
                ldb,
                beta,
                C,
                Ctype,
                ldc,
                batchCount,
                computeType,
                algo
            );
        }
    }

    retval=ssblasEx::cpu::GemmBatchedEx::ssblasGemmBatchedExSelector(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A,
        Atype,
        lda,
        B,
        Btype,
        ldb,
        beta,
        C,
        Ctype,
        ldc,
        batchCount,
        computeType,
        algo
    );
    return retval;
}
ssblasStatus_t ssblasGemmBatchedEx(
    ssblasOperation_t    transa,
    ssblasOperation_t    transb,
    SSINT                  m,
    SSINT                  n,
    SSINT                  k,
    const void*          alpha,
    const void *const    A[],
    ssblasDataType_t     Atype,
    const SSINT            lda,
    const void *const    B[],
    ssblasDataType_t     Btype,
    const SSINT            ldb,
    const void*          beta,
    void *const          C[],
    ssblasDataType_t     Ctype,
    const SSINT            ldc,
    const SSINT            batchCount,
    ssblasComputeType_t  computeType,
    ssblasGemmAlgo_t     algo
)
{
    ssblasStatus_t retval;
    //retval = FJBLAS_STATUS_SUCCESS;

    retval=ssblasGemmEx_main(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A,
        Atype,
        lda,
        B,
        Btype,
        ldb,
        beta,
        C,
        Ctype,
        ldc,
        batchCount,
        computeType,
        algo
    );
    return retval;
}

ssblasStatus_t ssblasGemmBatchedEx_64(
    ssblasOperation_t    transa,
    ssblasOperation_t    transb,
    SSLONG                  m,
    SSLONG                  n,
    SSLONG                  k,
    const void*          alpha,
    const void *const    A[],
    ssblasDataType_t     Atype,
    const SSLONG            lda,
    const void *const    B[],
    ssblasDataType_t     Btype,
    const SSLONG            ldb,
    const void*          beta,
    void *const        C[],
    ssblasDataType_t     Ctype,
    const SSLONG            ldc,
    const SSLONG            batchCount,
    ssblasComputeType_t  computeType,
    ssblasGemmAlgo_t     algo
)
{
    ssblasStatus_t retval;
    //retval = FJBLAS_STATUS_SUCCESS;

    /*retval=ssblasGemmEx_main_64(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A,
        Atype,
        lda,
        B,
        Btype,
        ldb,
        beta,
        C,
        Ctype,
        ldc,
        computeType,
        algo
    );*/
    return retval;
}


//Simple Simd BLAS for Batched