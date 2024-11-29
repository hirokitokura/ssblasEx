#ifndef XGEMM_SCALE_KERNEL
#define XGEMM_SCALE_KERNEL
//#include "def_sve_asm.h"
//#include <arm_sve.h>


namespace ssblasEx{
namespace cpu{
namespace xgemm{
namespace simd{
namespace Cscale{

using namespace ssblasEx::cpu::utils;


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB,typename DEF_DOUBLE_C>
void Xgemm_scale_kernel_0
(
    const INDEXINT ROW,
    const INDEXINT COL,
    const T_SCALE alpha,
    DEF_DOUBLE_C* C,
    const INDEXINT ldc
)
{
    if(!(ROW>0)){
        return ;
    }
    if(!(COL>0)){
        return ;
    }

    DEF_DOUBLE_C *c_offset;
    const INDEXINT nvl=ssblasGemmBatchedEx_SIMDLENGTH<INDEXINT,DEF_DOUBLE_C>();
    const INDEXINT nul=nvl*COMPUTE_SIMDROW_NUM_IN_GEMM;
    PTRUE_PS(p0);
    //Z0: 
    //Z1: 
    //Z2: 
    //Z3: 
    //Z4: 0の値 
    DUP_ZSI(z4 ,0);

    //端数のpredicate計算
    WHILELT_PSX(p1,((ROW/nvl)*nvl),ROW);
    for(INDEXINT j=0;j<COL;j++){
        c_offset=&C[0+j*ldc];
        
        //4 SIMDずつ
        INDEXINT im=0;
        for(im=0; im+nul<=ROW;im+=nul){
            ST1W_ZXI(z4,p0,c_offset,0);
            ST1W_ZXI(z4,p0,c_offset,1);
            ST1W_ZXI(z4,p0,c_offset,2);
            ST1W_ZXI(z4,p0,c_offset,3);
            c_offset+=nul;
        }
        //1 SIMDずつ
        for(; im+nvl<=ROW;im+=nvl){
            ST1W_ZXI(z4,p0,c_offset,0);
            c_offset+=nvl;
        }

        //端数
        //1 SIMDずつ
        //for(; im+nvl<=ROW;im+=nvl)
        {
            ST1W_ZXI(z4,p1,c_offset,0);
        }
    }
}

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB,typename DEF_DOUBLE_C>
void Xgemm_scale_kernel_scalar
(
    const INDEXINT ROW,
    const INDEXINT COL,
    const T_SCALE alpha0,
    DEF_DOUBLE_C* C,
    const INDEXINT ldc
)
{
    if(!(ROW>0)){
        return ;
    }
    if(!(COL>0)){
        return ;
    }

    DEF_DOUBLE_C *c_offset;
    const INDEXINT nvl=ssblasGemmBatchedEx_SIMDLENGTH<INDEXINT,DEF_DOUBLE_C>();
    const INDEXINT nul=nvl*COMPUTE_SIMDROW_NUM_IN_GEMM;
    PTRUE_PS(p0);
    //Z0: 
    //Z1: 
    //Z2: 
    //Z3: 
    //Z4: alphaの値 

    T_SCALE alpha=alpha0;
	T_SCALE *alphap=&alpha;
    LD1RW_ZXI(z4,p0,alphap,0);

    //端数のpredicate計算
    WHILELT_PSX(p1,((ROW/nvl)*nvl),ROW);
    for(INDEXINT j=0;j<COL;j++){
        c_offset=&C[0+j*ldc];
        
        //4 SIMDずつ
        INDEXINT im=0;
        for(im=0; im+nul<=ROW;im+=nul){
            LD1W_ZXI(z0,p0,c_offset,0);
            LD1W_ZXI(z1,p0,c_offset,1);
            LD1W_ZXI(z2,p0,c_offset,2);
            LD1W_ZXI(z3,p0,c_offset,3);

            FMUL_ZSP_base(z0, p0, z4);
            FMUL_ZSP_base(z1, p0, z4);
            FMUL_ZSP_base(z2, p0, z4);
            FMUL_ZSP_base(z3, p0, z4);

            ST1W_ZXI(z0,p0,c_offset,0);
            ST1W_ZXI(z1,p0,c_offset,1);
            ST1W_ZXI(z2,p0,c_offset,2);
            ST1W_ZXI(z3,p0,c_offset,3);
            c_offset+=nul;
        }
        //1 SIMDずつ
        for(; im+nvl<=ROW;im+=nvl){
            LD1W_ZXI(z0,p0,c_offset,0);
            FMUL_ZSP_base(z0, p0, z4);
            ST1W_ZXI(z0,p0,c_offset,0);
            c_offset+=nvl;
        }

        //端数
        //1 SIMDずつ
        //for(; im+nvl<=ROW;im+=nvl)
        {
            LD1W_ZXI(z0,p1,c_offset,0);
            FMUL_ZSP_base(z0, p1, z4);
            ST1W_ZXI(z0,p1,c_offset,0);
        }
    }
}
    
template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB,typename DEF_DOUBLE_C>
void Xgemm_scale_kernel
(
    const INDEXINT ROW,
    const INDEXINT COL,
    const T_SCALE alpha,
    DEF_DOUBLE_C* C,
    const INDEXINT ldc
)
{
    if(!(ROW>0)){
        return ;
    }
    if(!(COL>0)){
        return ;
    }

    if(((T_SCALE)0)==alpha){
        Xgemm_scale_kernel_0<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
        (
            ROW, COL,
            alpha,
            C, ldc
        );
        /*for(INDEXINT j=0;j<COL;j++){
            for(INDEXINT i=0;i<ROW;i++){
                C[i+j*ldc]=(T_SCALE)0;
            }
        }*/
    }else{
        Xgemm_scale_kernel_scalar<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
        (
            ROW, COL,
            alpha,
            C, ldc
        );
        /*for(INDEXINT j=0;j<COL;j++){
            for(INDEXINT i=0;i<ROW;i++){
                C[i+j*ldc]*=alpha;
            }
        }*/
    }
}

}
}
}
}
}
#endif