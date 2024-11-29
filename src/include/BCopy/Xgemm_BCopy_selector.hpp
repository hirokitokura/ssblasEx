#ifndef XGEMM_BCOPY_SELECTOR
#define XGEMM_BCOPY_SELECTOR

#include"Xgemm_ncopy_sve_1.hpp"
#include"Xgemm_tcopy_sve_1.hpp"

namespace ssblasEx{
namespace cpu{
namespace xgemm{
namespace simd{
namespace Bcopy{

template<typename INDEXINT, typename T_SCALE,  typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_BCopy_selector(INDEXINT m, INDEXINT n, const ssblasOperation_t DEF_TRANS, const DEF_DOUBLE_AB * a, INDEXINT lda, DEF_DOUBLE_AB *b, INDEXINT ldbl){
  //Xgemm_ncopy_sve_1<INDEXINT, T_SCALE, float, float>(m, n, a, lda, b);
  if constexpr (std::is_same_v<DEF_DOUBLE_AB, float>||std::is_same_v<DEF_DOUBLE_AB, double>) {
        if  (SSBLAS_OP_N==DEF_TRANS) {
            Xgemm_ncopy_sve_1<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_AB>(m, n, a, lda, b, ldbl);
        }
        else if  (SSBLAS_OP_T==DEF_TRANS) {
            Xgemm_tcopy_sve_1<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_AB>(n, m, a, lda, b, ldbl);
        }
        else if  (SSBLAS_OP_C==DEF_TRANS) {

        }
  }
  //else if constexpr (std::is_same_v<DEF_DOUBLE_AB, double>) Xgemm_ncopy_sve_1<INDEXINT, T_SCALE, double, double>(m, n, a, lda, b);
  else if constexpr (std::is_same_v<DEF_DOUBLE_AB, SSBLAS_SCHAR>){
        if  (SSBLAS_OP_N==DEF_TRANS) {
            Xgemm_ncopy_sve_v1_pack4<INDEXINT, T_SCALE, DEF_DOUBLE_AB, ssblaschar4>(m, n, a, lda, (ssblaschar4*)b, ldbl);
        }
        else if  (SSBLAS_OP_T==DEF_TRANS) {
            Xgemm_tcopy_sve_v1_pack4<INDEXINT, T_SCALE, DEF_DOUBLE_AB, ssblaschar4>(n, m, a, lda, (ssblaschar4*)b, ldbl);
        }
        else if  (SSBLAS_OP_C==DEF_TRANS) {

        }
  } 

}
template<typename INDEXINT, typename T_SCALE,  typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_BCopy(INDEXINT m, INDEXINT n, const ssblasOperation_t DEF_TRANS, const DEF_DOUBLE_AB * a, INDEXINT lda, DEF_DOUBLE_AB *b, INDEXINT ldbl){
    Xgemm_BCopy_selector<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>(m, n, DEF_TRANS, a, lda, b, ldbl);
}

}
}
}
}
}
#endif