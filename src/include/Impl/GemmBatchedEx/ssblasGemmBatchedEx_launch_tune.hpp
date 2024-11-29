#ifndef SSBLAS_LAUNCH_TUNE
#define SSBLAS_LAUNCH_TUNE

#include"../../ssblasGemmBatchedEx_functions.hpp"

#include"LAUNCHBLOCKSIZE_t.hpp"
#include"launch_kernels/ssblasGemmBatchedEx_Batchgemm_tune_kernel.hpp"

namespace ssblasEx{
namespace cpu{
namespace GemmBatchedEx{
namespace impl{
namespace simd{

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
ssblasStatus_t ssblasGemmBatchedEx_launch_tune(
    ssblasOperation_t    transa,
    ssblasOperation_t    transb,
    INDEXINT                  m,
    INDEXINT                  n,
    INDEXINT                  k,
    const T_SCALE          alpha,
    const DEF_DOUBLE_AB *const    A[],
    const INDEXINT            lda,
    const DEF_DOUBLE_AB *const    B[],
    const INDEXINT            ldb,
    const T_SCALE          beta,
    DEF_DOUBLE_C *const         C[],
    const INDEXINT            ldc,
    const INDEXINT            batchCount
)
{
    ssblasStatus_t retval=SSBLAS_STATUS_SUCCESS;
    using namespace ssblasEx::cpu::utils;
    INDEXINT THREAD_NUM;
    THREAD_NUM=(INDEXINT)ssblas_GetThreadNum();
    ssblascomputemode_t computemode;
    computemode=ssblas_get_computemode<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
                    (
                        transa,transb,
                        m,n,k,
                        lda, ldb, ldc,
                        batchCount,
                        THREAD_NUM
                    );
    if(SSBLAS_computemode_BMKN==computemode){
        retval=ssblasGemmBatchedEx_Batchgemm_tune_BMKN_host
        (
            transa,
            transb,
            m,n,k,
            alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc,
            batchCount
        );
    }else if(SSBLAS_computemode_BNKM==computemode){
        retval=ssblasGemmBatchedEx_Batchgemm_tune_BNKM_host
        (
            transa,
            transb,
            m,n,k,
            alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc,
            batchCount
        );
    }else{
        return SSBLAS_STATUS_INTERNAL_ERROR;
    }



    return retval;
}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
ssblasStatus_t ssblasGemmBatchedEx_launch_single(
    ssblasOperation_t    transa,
    ssblasOperation_t    transb,
    INDEXINT                  m,
    INDEXINT                  n,
    INDEXINT                  k,
    const T_SCALE          alpha,
    const DEF_DOUBLE_AB *const    A[],
    const INDEXINT            lda,
    const DEF_DOUBLE_AB *const    B[],
    const INDEXINT            ldb,
    const T_SCALE          beta,
    DEF_DOUBLE_C *const         C[],
    const INDEXINT            ldc,
    const INDEXINT            batchCount
)
{
    ssblasStatus_t retval=SSBLAS_STATUS_SUCCESS;

    retval=ssblasGemmBatchedEx_Batchgemm_single_BMNK_host
    (
        transa,
        transb,
        m,n,k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc,
        batchCount
    );


    return retval;
}

}
}
}
}
}

#endif