#ifndef XGEMM_CSCALE_SELECTOR
#define XGEMM_CSCALE_SELECTOR


#include"Xgemm_scale_kernel.hpp"

namespace ssblasEx{
namespace cpu{
namespace xgemm{
namespace simd{
namespace Cscale{

template<typename INDEXINT, typename T_SCALE,typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_CScale_selector
(
    const INDEXINT ROW,
    const INDEXINT COL,
    const T_SCALE alpha,
    DEF_DOUBLE_C* C,
    const INDEXINT ldc
)
{

    Xgemm_scale_kernel<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
    (
        ROW,
        COL,
        alpha,
        C,
        ldc
    );
}

template<typename INDEXINT, typename T_SCALE,typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_CScale
(
    const INDEXINT ROW,
    const INDEXINT COL,
    const T_SCALE alpha,
    DEF_DOUBLE_C* C,
    const INDEXINT ldc
)
{
    Xgemm_CScale_selector<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
    (
        ROW,
        COL,
        alpha,
        C,
        ldc
    );
}





}
}
}
}
}
#endif