#ifndef XGEMM_TCOPY_SVE_1
#define XGEMM_TCOPY_SVE_1
//#include "def_sve_asm.h"
//#include <arm_sve.h>

namespace ssblasEx{
namespace cpu{
namespace xgemm{
namespace simd{
namespace Bcopy{

using namespace ssblasEx::cpu::utils;


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
int Xgemm_tcopy_sve_1(INDEXINT m, INDEXINT n, const DEF_DOUBLE_AB * a, INDEXINT lda, DEF_DOUBLE_AB *b, INDEXINT ldb){
	//INDEXINT ldb=n;
	INDEXINT in,im,iin,iim,nii,mii;
	const static INDEXINT NB=32;
#define A(im,in) a[(im)+(in)*lda]
#define B(in,im) b[(in)+(im)*ldb]

	if(m==5) {
		for(in=0;in<n;in++) {
			__builtin_prefetch(&(A(9,in)),0,2);
			B(in,0)=A(0,in);
			B(in,1)=A(1,in);
			B(in,2)=A(2,in);
			B(in,3)=A(3,in);
			B(in,4)=A(4,in);
		}
		return 0;
	}
	for(in=0;in<n-NB+1;in+=NB) {
		for(im=0;im<m-NB+1;im+=NB) {
			for(iin=0;iin<NB;iin++) {
				__builtin_prefetch(&(A(im+iim   ,in+iin+16)),0,2);
				__builtin_prefetch(&(A(im+iim+31,in+iin+16)),0,2);
				__builtin_prefetch(&(A(im+iim   ,in+iin+2)),0,0);
				__builtin_prefetch(&(A(im+iim+31,in+iin+2)),0,0);
				for(iim=0;iim<NB;iim++) {
					B(in+iin,im+iim)=A(im+iim,in+iin);
				}
			}
		}
		mii=im;
		for(im=mii;im</*=*/m;im++) {
			for(iin=0;iin<NB;iin++) {
				B(in+iin,im)=A(im,in+iin);
			}

		}
	}

	nii=in;
	for(im=0;im<m-NB+1;im+=NB) {
		for(in=nii;in<n;in++) {
			if(n-nii>16) {
				__builtin_prefetch(&(A(im+iim   ,in+iin+16)),0,2);
				__builtin_prefetch(&(A(im+iim+31,in+iin+16)),0,2);
			}
			__builtin_prefetch(&(A(im+iim   ,in+iin+2)),0,0);
			__builtin_prefetch(&(A(im+iim+31,in+iin+2)),0,0);
			for(iim=0;iim<NB;iim++) {
				B(in,im+iim)=A(im+iim,in);
			}
		}
	}
	mii=im;
	for(iim=mii;iim<m;iim++) {
		for(in=nii;in<n;in++) {
			B(in,iim)=A(iim,in);
		}
	}
	return 0;

    #undef A
    #undef B
}

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
int Xgemm_tcopy_sve_v1_pack4(INDEXINT m, INDEXINT n, const DEF_DOUBLE_AB * a, INDEXINT lda, DEF_DOUBLE_C *b, INDEXINT ldb)
{
    constexpr INDEXINT PACK_SIZE=4;

    for(INDEXINT input_row_index=0;input_row_index<m;input_row_index++){
        for(INDEXINT input_col_index=0;input_col_index<n;input_col_index+=PACK_SIZE){
            ssblaschar4 tmp;
            tmp.x=a[(input_row_index+0)+(input_col_index+0)*lda];
            tmp.y=(input_col_index+1)<n?a[(input_row_index+0)+(input_col_index+1)*lda]:0;
            tmp.z=(input_col_index+2)<n?a[(input_row_index+0)+(input_col_index+2)*lda]:0;
            tmp.w=(input_col_index+3)<n?a[(input_row_index+0)+(input_col_index+3)*lda]:0;

            b[(input_col_index/PACK_SIZE)+(input_row_index)*ldb]=tmp;
        }
    }

    return 0;
}

}
}
}
}
}
#endif
