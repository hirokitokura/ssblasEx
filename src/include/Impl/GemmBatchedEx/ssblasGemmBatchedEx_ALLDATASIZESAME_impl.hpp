#include<iostream>

#include"../../ssblasGemmBatchedEx_functions.hpp"

#include"ssblasGemmBatchedEx_launch_tune.hpp"

namespace ssblasEx{
namespace cpu{
namespace GemmBatchedEx{
namespace impl{
namespace simd{

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
ssblasStatus_t ssblasGemmBatchedEx_ALLDATASIZESAME_NAIVE_TUNE
(
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
    using namespace ssblasEx::cpu::utils;
    ssblasStatus_t retval;
    retval=SSBLAS_STATUS_NOT_COMPUTED_ERROR;

    INDEXINT THREAD_NUM;
    THREAD_NUM=(INDEXINT)ssblas_GetThreadNum();
    //スレッド数が2,3,4,6,12の倍数以外の場合は処理を終了する。
    if(ssblas_acceptable_threadnum_for_tuneroute(THREAD_NUM)==false){
        retval=SSBLAS_STATUS_NOT_COMPUTED_ERROR;
        return retval;
    }
    //printf("ssblas_acceptable_threadnum_for_tuneroute end\n");

    ssblascomputemode_t computemode;
    computemode=ssblas_get_computemode<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
                    (
                        transa,transb,
                        m,n,k,
                        lda, ldb, ldc,
                        batchCount,
                        THREAD_NUM
                    );
    //printf("ssblas_get_computemode end\n");

    retval=ssblasGemmBatchedEx_launch_tune
    <INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
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
            batchCount
        );

    //retval=SSBLAS_STATUS_COMPUTED;
    //retval=SSBLAS_STATUS_NOT_COMPUTED;
    return retval;
}

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
ssblasStatus_t ssblasGemmBatchedEx_ALLDATASIZESAME_NAIVE_ALL_NAIVE
(
    ssblasOperation_t    transa,
    ssblasOperation_t    transb,
    const INDEXINT                  m,
    const INDEXINT                  n,
    const INDEXINT                  k,
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
    ssblasStatus_t retval;
    retval=SSBLAS_STATUS_NOT_COMPUTED_ERROR;
    retval=ssblasGemmBatchedEx_launch_single
    <INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
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
        batchCount
    );
    return retval;
}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
ssblasStatus_t ssblasGemmBatchedEx_ALLDATASIZESAME_NAIVE_NN(
    ssblasOperation_t    transa,
    ssblasOperation_t    transb,
    INDEXINT                  m,
    INDEXINT                  n,
    INDEXINT                  k,
    const void*          alpha_void,
    const void *const    A_void[],
    const INDEXINT            lda,
    const void *const    B_void[],
    const INDEXINT            ldb,
    const void*          beta_void,
    void *const         C_void[],
    const INDEXINT            ldc,
    const INDEXINT            batchCount
)
{
    ssblasStatus_t retval=SSBLAS_STATUS_INTERNAL_ERROR;
    const DEF_DOUBLE_AB *const *A = (const DEF_DOUBLE_AB *const *)A_void;  // ポインタのキャストを修正
    const DEF_DOUBLE_AB *const *B = (const DEF_DOUBLE_AB *const *)B_void;  // ポインタのキャストを修正
    DEF_DOUBLE_C **C = (DEF_DOUBLE_C **)C_void;  // ポインタのキャストを修正

    const T_SCALE alpha=*(const T_SCALE*)alpha_void;
    const T_SCALE beta=*(const T_SCALE*)beta_void;
    ssblasinternalComputeStatus_t internal_status;


    /*入力パラメータチェック*/
    /*不適切なパラメータなら SSBLAS_STATUS_INVALID_VALUE*/
    /*m,n,k,batchCount,alpha,betaに応じて何もせずreturn SSBLAS_STATUS_SUCCESS;*/
    /*Cのスケールのみを行ってreturn SSBLAS_STATUS_SUCCESS;*/


    /*特定のスレッド数に特化したチューニングルート*/
    retval=ssblasGemmBatchedEx_ALLDATASIZESAME_NAIVE_TUNE
    <INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
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
        batchCount
    );

    if(SSBLAS_STATUS_SUCCESS==retval){
        //printf("ssblasGemmBatchedEx_ALLDATASIZESAME_NAIVE_TUNE_MNPARA: SSBLAS_STATUS_COMPUTED\n");
        return retval;
    }
    //printf("ssblasGemmBatchedEx_ALLDATASIZESAME_NAIVE_ALL_NAIVE start\n");
    //ナイーブな実装
    retval=ssblasGemmBatchedEx_ALLDATASIZESAME_NAIVE_ALL_NAIVE
    <INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
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
        batchCount
    );
    if(SSBLAS_STATUS_SUCCESS==retval){
        //printf("ssblasGemmBatchedEx_ALLDATASIZESAME_NAIVE_ALL_NAIVE: SSBLAS_STATUS_COMPUTED\n");
        return retval;
    }

    return retval;
}

}
}
}
}
}
