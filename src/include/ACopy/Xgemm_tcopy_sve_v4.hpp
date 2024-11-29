/***************************************************************************
(c) RIKEN 2024, 2024. All rights reserved. sgemm_ncopy_sve_v4.c 0.3.26
Copyright 2017,2024 FUJITSU limited
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
   3. Neither the name of the OpenBLAS project nor the names of
      its contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

#ifndef XGEMM_TCOPY_SVE_V4
#define XGEMM_TCOPY_SVE_V4
//#include "def_sve_asm.h"
//#include <arm_sve.h>
namespace ssblasEx{
namespace cpu{
namespace xgemm{
namespace simd{
namespace Acopy{
using namespace ssblasEx::cpu::utils;

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_tcopy_sve_v4_core(INDEXINT k, INDEXINT m, const DEF_DOUBLE_AB *a, INDEXINT lda, DEF_DOUBLE_AB *aw)
{
#define A(ik,im) a[(ik)+(im)*lda]
	INDEXINT im,ik,iaw,ii;
	INDEXINT imb,kk,mm,mr,mbr,kbr;
	INDEXINT mbw;
	INDEXINT nvl=ssblasGemmBatchedEx_SIMDLENGTH<INDEXINT,DEF_DOUBLE_AB>();
	INDEXINT nsi=4,mb=nvl*nsi;

	for(imb=0;imb<m-mb+1;imb+=mb) {
		for(ik=0;ik<k-mb+1;ik+=mb) {
			iaw=k*imb  +ik*mb;
			for(im=0;im<mb;im++) {
				if(imb+im+mb<m) {
					__builtin_prefetch(&(A(ik+ii,imb+im+mb)),0,2);
					__builtin_prefetch(&(A(ik+ii+31,imb+im+mb)),0,2);
				}
				__builtin_prefetch(&(A(ik+ii,imb+im)),0,0);
				__builtin_prefetch(&(A(ik+ii+31,imb+im)),0,0);
				for(ii=0;ii<mb;ii++) {
					aw[iaw+mb*ii]=A(ik+ii,imb+im);
				}
				iaw=iaw+1;
			}
		}
		kk=ik;
		for(im=0;im<mb;im++) {
			iaw=im+k*imb+kk*mb;
			if(imb+im+mb<m) {
				__builtin_prefetch(&(A(kk,imb+im+mb)),0,2);
			}
			__builtin_prefetch(&(A(kk,imb+im)),0,0);
			for(ik=kk;ik<k;ik++) {
				aw[iaw]=A(ik,imb+im);
				iaw=iaw+mb;
			}
		}
	}
	mbr=m%mb;
	mm=m-mbr;
	mr=mbr;
	if(mr!=0) {
		for(ik=0;ik<k-mb+1;ik+=mb) {
			iaw=k*mm +ik*mr;
			for(im=mm;im<mm+mr;im++) {
				if(imb+im+mb<m) {
					__builtin_prefetch(&(A(ik+ii,imb+mb)),0,2);
					__builtin_prefetch(&(A(ik+ii+31,imb+mb)),0,2);
				}
				__builtin_prefetch(&(A(ik+ii,im-1)),0,0);
				__builtin_prefetch(&(A(ik+ii+31,im-1)),0,0);
				for(ii=0;ii<mb;ii++) {
					aw[iaw+mr*ii]=A(ik+ii,im);
				}
				iaw=iaw+1;
			}
		}
		kbr=k%mb;
		kk=k-kbr;
		iaw=k*mm+kk*mr;
		for(ik=kk;ik<k;ik++) {
			for(im=mm;im<mm+mr;im++) {
				aw[iaw]=A(ik,im);
				iaw=iaw+1;
			}
		}
	}
	
	#undef A
}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_tcopy_sve_v4(
	INDEXINT k, INDEXINT m, 
	const DEF_DOUBLE_AB *a, INDEXINT lda, 
	DEF_DOUBLE_AB *aw, 
	const INDEXINT TEAM_SIZE, const INDEXINT TEAM_ID
)
{
	const INDEXINT BASE_M_SIZE=COMPUTE_SIMDROW_NUM_IN_GEMM*ssblasGemmBatchedEx_SIMDLENGTH<INDEXINT,DEF_DOUBLE_AB>();
	const INDEXINT BASE_K_SIZE=64;

	const INDEXINT BASE_K_BLOCK_SIZE=BASE_M_SIZE*BASE_K_SIZE;
	const INDEXINT BASE_M_BLOCK_SIZE=BASE_M_SIZE*k;

	INDEXINT JOB_M_NUM=(m+BASE_M_SIZE-1)/BASE_M_SIZE;
	INDEXINT JOB_K_NUM=(k+BASE_K_SIZE-1)/BASE_K_SIZE;


	for(INDEXINT job_index=TEAM_ID;job_index<JOB_M_NUM*JOB_K_NUM;job_index+=TEAM_SIZE){
		const INDEXINT M_JOB_INDEX=(job_index%JOB_M_NUM);
		const INDEXINT K_JOB_INDEX=((job_index/JOB_M_NUM)%JOB_K_NUM);
		const INDEXINT START_M_INDEX=M_JOB_INDEX*BASE_M_SIZE;
		const INDEXINT START_K_INDEX=K_JOB_INDEX*BASE_K_SIZE;

		INDEXINT remain_m;
		INDEXINT remain_k;
		remain_m=START_M_INDEX+BASE_M_SIZE<m?BASE_M_SIZE:m-START_M_INDEX;
		remain_k=START_K_INDEX+BASE_K_SIZE<k?BASE_K_SIZE:k-START_K_INDEX;

		INDEXINT ldlocal;
		ldlocal=START_M_INDEX+BASE_M_SIZE<m?(BASE_M_SIZE):(m-START_M_INDEX);
		Xgemm_tcopy_sve_v4_core
		<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
		(
			remain_k,
			remain_m,
			&a[START_M_INDEX*lda+START_K_INDEX], lda,
			&aw[
				ldlocal*START_K_INDEX
				+(M_JOB_INDEX*BASE_M_BLOCK_SIZE)
				]
		);

	}
	return ;
	/*if(TEAM_ID==0){
		Xgemm_tcopy_sve_v4_core
		<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
		(
			k, m, 
			a, lda,
			aw
		);
	}*/


}

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_tcopy_sve_v4_pack4_core(INDEXINT k, INDEXINT m, const DEF_DOUBLE_AB *a, INDEXINT lda, DEF_DOUBLE_C *aw)
{

	constexpr INDEXINT PACK_SIZE=4;
	const INDEXINT SIMD_I32_NUM=ssblasGemmBatchedEx_SIMDLENGTH<INDEXINT,DEF_DOUBLE_AB>()/PACK_SIZE;
    const INDEXINT FJBLAS_BASE_MSIMD=SIMD_I32_NUM*COMPUTE_SIMDROW_NUM_IN_GEMM;
	for(INDEXINT input_row_index=0;input_row_index<k;input_row_index+=PACK_SIZE){
		INDEXINT input_col_index=0;
		//4SIMD単位の範囲
        for(input_col_index=0;input_col_index+FJBLAS_BASE_MSIMD<=m;input_col_index+=FJBLAS_BASE_MSIMD){
			for(INDEXINT i=0;i<FJBLAS_BASE_MSIMD;i++){
			ssblaschar4 tmp;
			tmp.x=a[(input_row_index+0)+(input_col_index+i)*lda];
			tmp.y=(input_row_index+1)<k?a[(input_row_index+1)+(input_col_index+i)*lda]:0;
			tmp.z=(input_row_index+2)<k?a[(input_row_index+2)+(input_col_index+i)*lda]:0;
			tmp.w=(input_row_index+3)<k?a[(input_row_index+3)+(input_col_index+i)*lda]:0;

			aw[((input_col_index+i)%FJBLAS_BASE_MSIMD)
			+(input_row_index/PACK_SIZE)*FJBLAS_BASE_MSIMD
			+FJBLAS_BASE_MSIMD*((k+PACK_SIZE-1)/PACK_SIZE)*((input_col_index+i)/FJBLAS_BASE_MSIMD)
			]=tmp;
			}
		}

		const INDEXINT ldaw=m-input_col_index;
				//((m-input_col_index)+PACK_SIZE-1)/PACK_SIZE;
		//端数
		for(;input_col_index<m;input_col_index+=1){
			ssblaschar4 tmp;
			tmp.x=a[(input_row_index+0)+(input_col_index+0)*lda];
			tmp.y=(input_row_index+1)<k?a[(input_row_index+1)+(input_col_index+0)*lda]:0;
			tmp.z=(input_row_index+2)<k?a[(input_row_index+2)+(input_col_index+0)*lda]:0;
			tmp.w=(input_row_index+3)<k?a[(input_row_index+3)+(input_col_index+0)*lda]:0;

			aw[
				((input_col_index)%FJBLAS_BASE_MSIMD)
				//+input_row_index*((((m+PACK_SIZE-1)/PACK_SIZE)%FJBLAS_BASE_MSIMD))
				+(input_row_index/PACK_SIZE)*ldaw/*(m-FJBLAS_BASE_MSIMD*((input_col_index/PACK_SIZE)/FJBLAS_BASE_MSIMD))*/
				+FJBLAS_BASE_MSIMD*((k+PACK_SIZE-1)/PACK_SIZE)*((input_col_index)/FJBLAS_BASE_MSIMD)
			]=tmp;
		}
	}
	
}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_tcopy_sve_v4_pack4(
	INDEXINT k, INDEXINT m, 
	const DEF_DOUBLE_AB *a, INDEXINT lda, 
	DEF_DOUBLE_C *aw, 
	const INDEXINT TEAM_SIZE, const INDEXINT TEAM_ID
)
{
	/*if(TEAM_ID==0){
		Xgemm_tcopy_sve_v4_pack4_core
		<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
		(
			k, m, 
			a, lda,
			aw
		);
	}*/
	const INDEXINT SSBLAS_K_PACK_SIZE=sizeof(DEF_DOUBLE_C)/sizeof(DEF_DOUBLE_AB);
	const INDEXINT BASE_M_SIZE=4*ssblasGemmBatchedEx_SIMDLENGTH<INDEXINT,DEF_DOUBLE_AB>()/SSBLAS_K_PACK_SIZE;
	const INDEXINT BASE_K_SIZE=64;

	const INDEXINT BASE_K_BLOCK_SIZE=BASE_M_SIZE*BASE_K_SIZE;
	const INDEXINT BASE_M_BLOCK_SIZE=(BASE_M_SIZE)* ((k+SSBLAS_K_PACK_SIZE-1)/SSBLAS_K_PACK_SIZE) *SSBLAS_K_PACK_SIZE;//DEF_DOUBLE_AB基準

	INDEXINT JOB_M_NUM=(m+BASE_M_SIZE-1)/BASE_M_SIZE;
	INDEXINT JOB_K_NUM=(k+BASE_K_SIZE-1)/BASE_K_SIZE;

	DEF_DOUBLE_AB* aw_=(DEF_DOUBLE_AB*)aw;
	for(INDEXINT job_index=TEAM_ID;job_index<JOB_M_NUM*JOB_K_NUM;job_index+=TEAM_SIZE){
		
		const INDEXINT M_JOB_INDEX=(job_index%JOB_M_NUM);
		const INDEXINT K_JOB_INDEX=((job_index/JOB_M_NUM)%JOB_K_NUM);
		const INDEXINT START_M_INDEX=M_JOB_INDEX*BASE_M_SIZE;
		const INDEXINT START_K_INDEX=K_JOB_INDEX*BASE_K_SIZE;

		INDEXINT remain_m;
		INDEXINT remain_k;
		remain_m=START_M_INDEX+BASE_M_SIZE<m?BASE_M_SIZE:m-START_M_INDEX;
		remain_k=START_K_INDEX+BASE_K_SIZE<k?BASE_K_SIZE:k-START_K_INDEX;

		INDEXINT ldlocal;
		//ldlocal=START_M_INDEX+BASE_M_SIZE<m?((BASE_M_SIZE+SSBLAS_K_PACK_SIZE-1)/SSBLAS_K_PACK_SIZE)*SSBLAS_K_PACK_SIZE:(((m-START_M_INDEX)+SSBLAS_K_PACK_SIZE-1)/SSBLAS_K_PACK_SIZE)*SSBLAS_K_PACK_SIZE;
		ldlocal=START_M_INDEX+BASE_M_SIZE<m?BASE_M_SIZE:(m-START_M_INDEX);

		/*printf("TEAM_ID: %d\n",TEAM_ID);
		printf("M_JOB_INDEX: %d\n",M_JOB_INDEX);
		printf("K_JOB_INDEX: %d\n",K_JOB_INDEX);
		printf("START_M_INDEX: %d\n",START_M_INDEX);
		printf("START_K_INDEX: %d\n",START_K_INDEX);
		printf("remain_m: %d\n",remain_m);
		printf("remain_k: %d\n",remain_k);
		printf("ldlocal: %d\n",ldlocal);
		printf("BASE_M_BLOCK_SIZE: %d\n",BASE_M_BLOCK_SIZE);*/
		Xgemm_tcopy_sve_v4_pack4_core
		<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
		(
			remain_k, remain_m,
			&a[START_M_INDEX*lda+START_K_INDEX], lda,
			(DEF_DOUBLE_C*)&aw_[
				ldlocal*(START_K_INDEX)
				+M_JOB_INDEX*BASE_M_BLOCK_SIZE]
		);
	}
}

}
}
}
}
}

#endif