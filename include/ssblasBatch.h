#ifndef SSBLASBATCH
#define SSBLASBATCH



#ifdef __cplusplus
extern "C" 
{
#endif

#include "ssblasBatch_type.h"



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
);

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
);
#ifdef __cplusplus
}
#endif

#endif