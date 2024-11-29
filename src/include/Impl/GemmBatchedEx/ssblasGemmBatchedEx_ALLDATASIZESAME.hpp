#include<iostream>
#include "../../../../include/ssblasBatch.h"
#include "./ssblasGemmBatchedEx_ALLDATASIZESAME_impl.hpp"

namespace ssblasEx{
namespace cpu{
namespace GemmBatchedEx{
namespace impl{


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
ssblasStatus_t ssblasGemmBatchedEx_ALLDATASIZESAME(
    ssblasOperation_t    transa,
    ssblasOperation_t    transb,
    INDEXINT                  m,
    INDEXINT                  n,
    INDEXINT                  k,
    const void*          alpha,
    const void *const    A[],
    const INDEXINT            lda,
    const void *const    B[],
    const INDEXINT            ldb,
    const void*          beta,
    void *const          C[],
    const INDEXINT            ldc,
    const INDEXINT            batchCount,
    ssblasComputeType_t  computeType,
    ssblasGemmAlgo_t     algo
)
{

    //将来mmlaやsmeが実装された場合、この関数でルートを切り替える
    //この時、mmlaやsme用の名前空間で実装を分ける
    //ssblasGemmAlgo_tで切り替える
    //SSBLAS_COMPUTE_DEFAULTの場合: そのCPUで最高性能を持つ実装を選択する
    //そのほかの場合: 指令された(simd, mmla, smeルートを選択する)

    //将来mmlaやsmeが実装された場合、using namespaceの範囲を修正する。
    using namespace ssblasEx::cpu::GemmBatchedEx::impl;
    
    ssblasStatus_t retval;
    //const DEF_DOUBLE alpha=*(const DEF_DOUBLE*)alpha_void;
    //const DEF_DOUBLE beta=*(const DEF_DOUBLE*)beta_void;
    retval=SSBLAS_STATUS_INTERNAL_ERROR;
    if((typeid(T_SCALE)==typeid(float))&&(typeid(DEF_DOUBLE_AB)==typeid(float))&&(typeid(DEF_DOUBLE_C)==typeid(float))){
        if(SSBLAS_COMPUTE_32F!=computeType){
            //retval=SSBLAS_STATUS_INVALID_VALUE;
            //return retval;
        }
        if(
            ((SSBLAS_OP_N==transa)&&(SSBLAS_OP_N==transb))
            ||((SSBLAS_OP_N==transa)&&(SSBLAS_OP_T==transb))
            ||((SSBLAS_OP_T==transa)&&(SSBLAS_OP_N==transb))
            ||((SSBLAS_OP_T==transa)&&(SSBLAS_OP_T==transb))
        ){
            retval=simd::ssblasGemmBatchedEx_ALLDATASIZESAME_NAIVE_NN
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
        /*else if((SSBLAS_OP_N==transa)&&(SSBLAS_OP_T==transb)){
            return SSBLAS_STATUS_NOTIMPLEMENTED_ERROR;
        }
        else if((SSBLAS_OP_T==transa)&&(SSBLAS_OP_N==transb)){
            return SSBLAS_STATUS_NOTIMPLEMENTED_ERROR;
        }
        else if((SSBLAS_OP_T==transa)&&(SSBLAS_OP_T==transb)){
            return SSBLAS_STATUS_NOTIMPLEMENTED_ERROR;
        }*/
        else{
            retval=SSBLAS_STATUS_INVALID_VALUE;
            return retval;
        }
    }
    else if((typeid(T_SCALE)==typeid(double))&&(typeid(DEF_DOUBLE_AB)==typeid(double))&&(typeid(DEF_DOUBLE_C)==typeid(double))){
        if(
            ((SSBLAS_OP_N==transa)&&(SSBLAS_OP_N==transb))
            ||((SSBLAS_OP_N==transa)&&(SSBLAS_OP_T==transb))
            ||((SSBLAS_OP_T==transa)&&(SSBLAS_OP_N==transb))
            ||((SSBLAS_OP_T==transa)&&(SSBLAS_OP_T==transb))
        ){
            retval=simd::ssblasGemmBatchedEx_ALLDATASIZESAME_NAIVE_NN
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
        else{
            retval=SSBLAS_STATUS_INVALID_VALUE;
            return retval;
        }
        return retval;
    }
    /*else if(typeid(DEF_DOUBLE)==typeid(_Float16)){
        retval=SSBLAS_STATUS_NOTIMPLEMENTED_ERROR;
        return retval;
    }*/
    else if((typeid(T_SCALE)==typeid(int))&&(typeid(DEF_DOUBLE_AB)==typeid(int))&&(typeid(DEF_DOUBLE_C)==typeid(int))){
        retval=SSBLAS_STATUS_NOTIMPLEMENTED_ERROR;
        return retval;
    }
    else if((typeid(T_SCALE)==typeid(int))&&(typeid(DEF_DOUBLE_AB)==typeid(SSBLAS_SCHAR))&&(typeid(DEF_DOUBLE_C)==typeid(int))){
        retval=//ssblasGemmBatchedEx_ALLDATASIZESAME_PACK4_DOT_NN
        simd::ssblasGemmBatchedEx_ALLDATASIZESAME_NAIVE_NN
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
        return retval;
    }
    else{
        retval=SSBLAS_STATUS_INVALID_VALUE;
        return retval;
    }
    return retval;
}


}
}
}
}
