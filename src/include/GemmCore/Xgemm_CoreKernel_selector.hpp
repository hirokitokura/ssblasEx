#ifndef XGEMM_COREKERNEL_SELECTOR
#define XGEMM_COREKERNEL_SELECTOR


#include"Xgemm_kernel_sve_v4x5_1.hpp"


namespace ssblasEx{
namespace cpu{
namespace xgemm{
namespace simd{
namespace GemmCore{

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void  Xgemm_CoreKernel_selector
(
	INDEXINT m,
	INDEXINT n,
	INDEXINT k,
	T_SCALE alpha0, 
	const DEF_DOUBLE_C* ap,
	const DEF_DOUBLE_C* b,
	INDEXINT ldb_,
	DEF_DOUBLE_C* cp, 
	INDEXINT ldc 
)
{



    Xgemm_kernel_sve_v4x5_1
    <INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
    (
        m,n,k,
        alpha0,
        ap,
        b, ldb_,
        cp, ldc
    );
}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void  Xgemm_CoreKernel
(
	INDEXINT m,
	INDEXINT n,
	INDEXINT k,
	T_SCALE alpha0, 
	const DEF_DOUBLE_C* ap,
	const DEF_DOUBLE_C* b,
	INDEXINT ldb_,
	DEF_DOUBLE_C* cp, 
	INDEXINT ldc 
)
{

    Xgemm_CoreKernel_selector<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
    (
        m,n,k,
        alpha0,
        ap,
        b, ldb_,
        cp, ldc
    );
}





}
}
}
}
}

#endif