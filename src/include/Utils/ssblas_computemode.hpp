#ifndef SSBLAS_COMPUTEMODE
#define SSBLAS_COMPUTEMODE

namespace ssblasEx{
namespace cpu{
namespace utils{

typedef enum {
  SSBLAS_computemode_BMKN, //行列Aを再利用ルート
  SSBLAS_computemode_BNKM, //行列Bを再利用ルート
  SSBLAS_computemode_Kreduction //行列積リダクションルート
} ssblascomputemode_t;


//チューニングルートにおける並列モード決定の関数。
template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
ssblascomputemode_t ssblas_get_computemode
(
    ssblasOperation_t    transa,
    ssblasOperation_t    transb,
    INDEXINT             m,
    INDEXINT             n,
    INDEXINT             k,
    const INDEXINT       lda,
    const INDEXINT       ldb,
    const INDEXINT       ldc,
    const INDEXINT       batchCount,
    INDEXINT             THREAD_NUM
)
{
    //Aをパックする場合、Aをパックのコストが大きいため、Mを再利用するルートを選択する。
    //パックしない場合は適宜、選択する
    ssblascomputemode_t retval=SSBLAS_computemode_BMKN;

    INDEXINT INNER_SIMD_SIZE=COMPUTE_SIMDROW_NUM_IN_GEMM*ssblasGemmBatchedEx_SIMDLENGTH<INDEXINT, DEF_DOUBLE_AB>();
    if constexpr (std::is_same_v<DEF_DOUBLE_AB, float>||std::is_same_v<DEF_DOUBLE_AB, double>){
      //暫定ルート
      if(m<=INNER_SIMD_SIZE*8){
        retval=SSBLAS_computemode_BMKN;
      }else{
        retval=SSBLAS_computemode_BNKM;
      }
    }
    else if constexpr (std::is_same_v<DEF_DOUBLE_AB, SSBLAS_SCHAR>){
      retval=SSBLAS_computemode_BMKN;
    }
    return retval;
}

}
}
}
#endif