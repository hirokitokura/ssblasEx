#include<iostream>
#include "../../../../include/ssblasBatch.h"
#include "ssblasGemmBatchedEx_ALLDATASIZESAME.hpp"

namespace ssblasEx{
namespace cpu{
namespace GemmBatchedEx{


ssblasStatus_t ssblasGemmBatchedExSelector(
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
    using namespace ssblasEx::cpu::GemmBatchedEx::impl;
    ssblasStatus_t retval;

    if((SSBLAS_R_8I==Atype)&&(SSBLAS_R_8I==Btype)&&(SSBLAS_R_32I==Ctype)){
        //retval=SSBLAS_STATUS_NOTIMPLEMENTED_ERROR;
        //return retval;
        retval=ssblasGemmBatchedEx_ALLDATASIZESAME
                <SSINT, int, SSBLAS_SCHAR, int>
                (
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    alpha,
                    A,
                    lda,
                    B,
                    ldb,
                    beta,
                    C,
                    ldc,
                    batchCount,
                    computeType,
                    algo  
                );
    }
    else if((SSBLAS_R_64F==Atype)&&(SSBLAS_R_64F==Btype)&&(SSBLAS_R_64F==Ctype)){
        //retval=SSBLAS_STATUS_NOTIMPLEMENTED_ERROR;
        //return retval;
        retval=ssblasGemmBatchedEx_ALLDATASIZESAME
                <SSINT, double, double, double>
                (
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    alpha,
                    A,
                    lda,
                    B,
                    ldb,
                    beta,
                    C,
                    ldc,
                    batchCount,
                    computeType,
                    algo  
                );
    }
    else if((SSBLAS_R_32F==Atype)&&(SSBLAS_R_32F==Btype)&&(SSBLAS_R_32F==Ctype)){
        retval=ssblasGemmBatchedEx_ALLDATASIZESAME
                <SSINT, float, float, float>
                (
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    alpha,
                    A,
                    lda,
                    B,
                    ldb,
                    beta,
                    C,
                    ldc,
                    batchCount,
                    computeType,
                    algo  
                );
    }
    else{
        retval=SSBLAS_STATUS_INVALID_VALUE;
    }

    return retval;
}


}
}
}