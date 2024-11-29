#ifndef XGEMM_ACOPY_SELECTOR
#define XGEMM_ACOPY_SELECTOR


#include"Xgemm_ncopy_sve_v4.hpp"
#include"Xgemm_tcopy_sve_v4.hpp"
namespace ssblasEx{
namespace cpu{
namespace xgemm{
namespace simd{
namespace Acopy{


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_ACopy_selector(
    INDEXINT k,INDEXINT m,
     const ssblasOperation_t DEF_TRANS, 
     const DEF_DOUBLE_AB *a,INDEXINT lda,
     DEF_DOUBLE_AB *aw,
     const INDEXINT TEAM_SIZE, const INDEXINT TEAM_ID
     )
{
	if constexpr (std::is_same_v<DEF_DOUBLE_AB, float>||std::is_same_v<DEF_DOUBLE_AB, double>){
        if  (SSBLAS_OP_N==DEF_TRANS){
            Xgemm_ncopy_sve_v4<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>(k, m, a, lda, aw, TEAM_SIZE, TEAM_ID);
        }
        else if  (SSBLAS_OP_T==DEF_TRANS){
            Xgemm_tcopy_sve_v4<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>(k, m, a, lda, aw, TEAM_SIZE, TEAM_ID);
        }
        else if  (SSBLAS_OP_C==DEF_TRANS){
        }
    }
	//else if constexpr (std::is_same_v<DEF_DOUBLE_AB, double>) Xgemm_tcopy_sve_v4<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>(k, m, a, lda, aw);
	//else if constexpr (std::is_same_v<DEF_DOUBLE_AB, int>) Xgemm_tcopy_sve_v4<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>(k, m, a, lda, aw);
	else if constexpr ((std::is_same_v<DEF_DOUBLE_AB, SSBLAS_SCHAR>)&&(std::is_same_v<DEF_DOUBLE_C, int>)){
        if  (SSBLAS_OP_N==DEF_TRANS){
            Xgemm_ncopy_sve_v4_pack4<INDEXINT, T_SCALE, DEF_DOUBLE_AB, ssblaschar4>(k, m, a, lda, /*(ssblaschar4*)*/aw, TEAM_SIZE, TEAM_ID);
        }
        else if  (SSBLAS_OP_T==DEF_TRANS){
            Xgemm_tcopy_sve_v4_pack4<INDEXINT, T_SCALE, DEF_DOUBLE_AB, ssblaschar4>(k, m, a, lda, (ssblaschar4*)aw, TEAM_SIZE, TEAM_ID);
        }
        else if  (SSBLAS_OP_C==DEF_TRANS){
        }

    } 

}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_ACopy(
    INDEXINT k,INDEXINT m,
    const ssblasOperation_t DEF_TRANS, 
    const DEF_DOUBLE_AB *a,INDEXINT lda,
    DEF_DOUBLE_AB *aw,
    const INDEXINT TEAM_SIZE, const INDEXINT TEAM_ID
)
{
    Xgemm_ACopy_selector<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>(k, m, DEF_TRANS, a, lda, aw, TEAM_SIZE, TEAM_ID);

}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_ACopy(
    INDEXINT k,INDEXINT m,
    const ssblasOperation_t DEF_TRANS, 
    const DEF_DOUBLE_AB *a,INDEXINT lda,
    DEF_DOUBLE_AB *aw
)
{
    Xgemm_ACopy<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
    (k, m, DEF_TRANS, a, lda, aw, 1, 0);

}

}
}
}
}
}
#endif