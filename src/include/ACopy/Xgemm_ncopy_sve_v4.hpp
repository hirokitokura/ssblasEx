/***************************************************************************
(c) RIKEN 2024, 2024. All rights reserved. sgemm_tcopy_sve_v4.c 0.3.26
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


#ifndef XGEMM_NCOPY_SVE_V4
#define XGEMM_NCOPY_SVE_V4
//#include "def_sve_asm.h"
//#include <arm_sve.h>
namespace ssblasEx{
namespace cpu{
namespace xgemm{
namespace simd{
namespace Acopy{

using namespace ssblasEx::cpu::utils;

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_ncopy_sve_v4_core(INDEXINT k,INDEXINT m,const DEF_DOUBLE_AB *a,INDEXINT lda,DEF_DOUBLE_AB *aw)
{
	//printf("Xgemm_tcopy_sve_v4\n");
	const DEF_DOUBLE_AB *ap0,*ap1;
	DEF_DOUBLE_AB *awp,*awp2;
	const DEF_DOUBLE_AB *apre0,*apre1;
	DEF_DOUBLE_AB *awpre;

	INDEXINT ik,j,im,ii;
	INDEXINT kb,iks,ike,kl,ipre;
	INDEXINT mm;

	const INDEXINT nvl=ssblasGemmBatchedEx_SIMDLENGTH<INDEXINT,DEF_DOUBLE_AB>();
	//printf("nvl: %d\n",nvl);
	PTRUE_PS(p0);

	const INDEXINT nul=COMPUTE_SIMDROW_NUM_IN_GEMM*nvl;

	kb=k;
	iks=0;
	kl=k;
	ike=k-1;

	ipre=lda*2;
	if(lda<=m+32) {
		ipre=32;
	}

	if(kl>0) {
		ap0=a+lda*iks;
		ap1=ap0+lda;
		apre0=ap0+lda*8;
		ik=iks;

		for(;ik<=ike-1;ik+=2) {
			im=0;
			awp=aw+nul*ik;
			if(m-im>=nul) {
				im+=nul;
				apre0=ap0+lda*4;
				apre1=ap1+lda*4;
				PRFM_XI(PLDL2STRM,apre0,0);
				PRFM_XI(PLDL2STRM,apre1,0);
				PRFM_XI(PLDL2STRM,apre0,256);
				PRFM_XI(PLDL2STRM,apre1,256);

				apre0=ap0+lda*2;
				apre1=ap1+lda*2;
				PRFM_XI(PLDL1STRM,apre0,0);
				PRFM_XI(PLDL1STRM,apre1,0);
				PRFM_XI(PLDL1STRM,apre0,256);
				PRFM_XI(PLDL1STRM,apre1,256);

				LD1W_ZXI(z0,p0,ap0,0);
				LD1W_ZXI(z1,p0,ap0,1);
				LD1W_ZXI(z2,p0,ap0,2);
				LD1W_ZXI(z3,p0,ap0,3);

				LD1W_ZXI(z8 ,p0,ap1,0);
				LD1W_ZXI(z9 ,p0,ap1,1);
				LD1W_ZXI(z10,p0,ap1,2);
				LD1W_ZXI(z11,p0,ap1,3);
				for(;im<m-nul+1;im+=nul) {
					ap0+=nul;
					ap1+=nul;

					apre0=ap0+lda*4;
					apre1=ap1+lda*4;
					PRFM_XI(PLDL2STRM,apre0,0);
					PRFM_XI(PLDL2STRM,apre1,0);

					ST1W_ZXI(z0,p0,awp,0);
					ST1W_ZXI(z1,p0,awp,1);
					ST1W_ZXI(z2,p0,awp,2);
					ST1W_ZXI(z3,p0,awp,3);
					awp2=awp+nul;

					ST1W_ZXI(z8,p0,awp2,0);
					ST1W_ZXI(z9,p0,awp2,1);
					ST1W_ZXI(z10,p0,awp2,2);
					ST1W_ZXI(z11,p0,awp2,3);
					awp+=k*nul;

					LD1W_ZXI(z0,p0,ap0,0);
					LD1W_ZXI(z1,p0,ap0,1);
					LD1W_ZXI(z2,p0,ap0,2);
					LD1W_ZXI(z3,p0,ap0,3);

					LD1W_ZXI(z8 ,p0,ap1,0);
					LD1W_ZXI(z9 ,p0,ap1,1);
					LD1W_ZXI(z10,p0,ap1,2);
					LD1W_ZXI(z11,p0,ap1,3);

					apre0=ap0+ipre;
					apre1=ap1+ipre;
					PRFM_XI(PLDL1STRM,apre0,0);
					PRFM_XI(PLDL1STRM,apre1,0);

					awp2=awp+64;
					PRFM_XI(PLDL1STRM,awp2,0);
					PRFM_XI(PLDL1STRM,awp2,256);

				}

				ST1W_ZXI(z0,p0,awp,0);
				ST1W_ZXI(z1,p0,awp,1);
				ST1W_ZXI(z2,p0,awp,2);
				ST1W_ZXI(z3,p0,awp,3);

				awp2=awp+nul;
				ST1W_ZXI(z8,p0,awp2,0);
				ST1W_ZXI(z9,p0,awp2,1);
				ST1W_ZXI(z10,p0,awp2,2);
				ST1W_ZXI(z11,p0,awp2,3);

				awp+=k*nul;
				ap0+=nul;
				ap1+=nul;
			}
			mm=m-im;
			awp=aw+k*(m-mm)+ik*mm;
			awp2=awp+mm;

			if(mm>=1) {
				if(mm>nvl*3) {
					j=nvl*3;
					WHILELT_PSX(p1,j,mm);

                                        apre0=ap0+lda*4;
                                        apre1=ap1+lda*4;
                                        PRFM_XI(PLDL2STRM,apre0,0);
                                        PRFM_XI(PLDL2STRM,apre1,0);

                                        apre0=ap0+lda*2;
                                        apre1=ap1+lda*2;
                                        PRFM_XI(PLDL1STRM,apre0,0);
                                        PRFM_XI(PLDL1STRM,apre1,0);

					LD1W_ZXI(z0,p0,ap0,0);
					LD1W_ZXI(z1,p0,ap1,0);
					LD1W_ZXI(z2,p0,ap0,1);
					LD1W_ZXI(z3,p0,ap1,1);
					LD1W_ZXI(z4,p0,ap0,2);
					LD1W_ZXI(z5,p0,ap1,2);
					LD1W_ZXI(z6,p1,ap0,3);
					LD1W_ZXI(z7,p1,ap1,3);


					ST1W_ZXI(z0,p0,awp ,0);
					ST1W_ZXI(z1,p0,awp2,0);
					ST1W_ZXI(z2,p0,awp ,1);
					ST1W_ZXI(z3,p0,awp2,1);
					ST1W_ZXI(z4,p0,awp ,2);
					ST1W_ZXI(z5,p0,awp2,2);
					ST1W_ZXI(z6,p1,awp ,3);
					ST1W_ZXI(z7,p1,awp2,3);

					ap0+=mm;
					ap1+=mm;

				}
				else if(mm>nvl*2) {
					j=nvl*2;
					WHILELT_PSX(p1,j,mm);

                                        apre0=ap0+lda*4;
                                        apre1=ap1+lda*4;
                                        PRFM_XI(PLDL2STRM,apre0,0);
                                        PRFM_XI(PLDL2STRM,apre1,0);

					LD1W_ZXI(z0,p0,ap0,0);
					LD1W_ZXI(z1,p0,ap1,0);
					LD1W_ZXI(z2,p0,ap0,1);
					LD1W_ZXI(z3,p0,ap1,1);
					LD1W_ZXI(z4,p1,ap0,2);
					LD1W_ZXI(z5,p1,ap1,2);

					ST1W_ZXI(z0,p0,awp ,0);
					ST1W_ZXI(z1,p0,awp2,0);
					ST1W_ZXI(z2,p0,awp ,1);
					ST1W_ZXI(z3,p0,awp2,1);
					ST1W_ZXI(z4,p1,awp ,2);
					ST1W_ZXI(z5,p1,awp2,2);

					ap0+=mm;
					ap1+=mm;
				}
				else if(mm>nvl) {
					j=nvl*1;
					WHILELT_PSX(p1,j,mm);

                                        apre0=ap0+lda*4;
                                        apre1=ap1+lda*4;
                                        PRFM_XI(PLDL2STRM,apre0,0);
                                        PRFM_XI(PLDL2STRM,apre1,0);

					LD1W_ZXI(z0,p0,ap0,0);
					LD1W_ZXI(z1,p0,ap1,0);
					LD1W_ZXI(z2,p1,ap0,1);
					LD1W_ZXI(z3,p1,ap1,1);

					ST1W_ZXI(z0,p0,awp ,0);
					ST1W_ZXI(z1,p0,awp2,0);
					ST1W_ZXI(z2,p1,awp ,1);
					ST1W_ZXI(z3,p1,awp2,1);

					ap0+=mm;
					ap1+=mm;
				}
				else if(mm>=1) {
					j=0;
					WHILELT_PSX(p1,j,mm);

                                        apre0=ap0+lda*4;
                                        apre1=ap1+lda*4;
                                        PRFM_XI(PLDL2STRM,apre0,0);
                                        PRFM_XI(PLDL2STRM,apre1,0);

					LD1W_ZXI(z0,p1,ap0,0);
					LD1W_ZXI(z1,p1,ap1,0);

					ST1W_ZXI(z0,p1,awp ,0);
					ST1W_ZXI(z1,p1,awp2,0);
					ap0+=mm;
					ap1+=mm;
				}

			}
			ap0+=lda*2-m;
			ap1+=lda*2-m;
		}
		for(;ik<=ike;ik++) {
			im=0;
			awp=aw+nul*ik;

			if(m-im>=nul) {
				im+=nul;
				apre0=ap0+ipre;
				PRFM_XI(PLDL1STRM,apre0,0);
				PRFM_XI(PLDL1STRM,apre0,256);

				LD1W_ZXI(z0,p0,ap0,0);
				LD1W_ZXI(z1,p0,ap0,1);
				LD1W_ZXI(z2,p0,ap0,2);
				LD1W_ZXI(z3,p0,ap0,3);
				for(;im<m-nul+1;im+=nul) {
					ap0+=nul;
					apre0=ap0+ipre;
					PRFM_XI(PLDL1STRM,apre0,0);
					PRFM_XI(PLDL1STRM,apre0,256);

					ST1W_ZXI(z0,p0,awp,0);
					ST1W_ZXI(z1,p0,awp,1);
					ST1W_ZXI(z2,p0,awp,2);
					ST1W_ZXI(z3,p0,awp,3);

					LD1W_ZXI(z0,p0,ap0,0);
					LD1W_ZXI(z1,p0,ap0,1);
					LD1W_ZXI(z2,p0,ap0,2);
					LD1W_ZXI(z3,p0,ap0,3);
					awp+=k*nul;
				}

				ST1W_ZXI(z0,p0,awp,0);
				ST1W_ZXI(z1,p0,awp,1);
				ST1W_ZXI(z2,p0,awp,2);
				ST1W_ZXI(z3,p0,awp,3);

				awp+=k*nul;
				ap0+=nul;
			}

			mm=m-im;
			awp=aw+k*(m-mm)+ik*mm;

			if(mm>nvl*3) {
				j=nvl*3;
				WHILELT_PSX(p1,j,mm);
				LD1W_ZXI(z0,p0,ap0,0);
				LD1W_ZXI(z2,p0,ap0,1);
				LD1W_ZXI(z4,p0,ap0,2);
				LD1W_ZXI(z6,p1,ap0,3);

				ST1W_ZXI(z0,p0,awp ,0);
				ST1W_ZXI(z2,p0,awp ,1);
				ST1W_ZXI(z4,p0,awp ,2);
				ST1W_ZXI(z6,p1,awp ,3);

				ap0+=mm;
			}
			else if(mm>nvl*2) {
				j=nvl*2;
				WHILELT_PSX(p1,j,mm);

				LD1W_ZXI(z0,p0,ap0,0);
				LD1W_ZXI(z2,p0,ap0,1);
				LD1W_ZXI(z4,p1,ap0,2);

				ST1W_ZXI(z0,p0,awp ,0);
				ST1W_ZXI(z2,p0,awp ,1);
				ST1W_ZXI(z4,p1,awp ,2);

				ap0+=mm;
			}
			else if(mm>nvl) {
				j=nvl*1;
				WHILELT_PSX(p1,j,mm);

				LD1W_ZXI(z0,p0,ap0,0);
				LD1W_ZXI(z1,p1,ap0,1);

				ST1W_ZXI(z0,p0,awp ,0);
				ST1W_ZXI(z1,p1,awp ,1);
				ap0+=mm;
			}
			else if(mm>=1) {
				j=0;
				WHILELT_PSX(p1,j,mm);

				LD1W_ZXI(z0,p1,ap0,0);

				ST1W_ZXI(z0,p1,awp,0);

				ap0+=mm;
			}

			ap0+=lda-m;
		}
	}
	//return 0;
}



template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_ncopy_sve_v4(
	INDEXINT k,INDEXINT m,
	const DEF_DOUBLE_AB *a,INDEXINT lda,
	DEF_DOUBLE_AB *aw, 
	const INDEXINT TEAM_SIZE, const INDEXINT TEAM_ID
){

	const INDEXINT BASE_M_SIZE=4*ssblasGemmBatchedEx_SIMDLENGTH<INDEXINT,DEF_DOUBLE_AB>();
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
		Xgemm_ncopy_sve_v4_core
		<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
		(
			remain_k,
			remain_m,
			&a[START_M_INDEX+START_K_INDEX*lda], lda,
			&aw[
				ldlocal*START_K_INDEX
				+(M_JOB_INDEX*BASE_M_BLOCK_SIZE)
				]
		);

	}
	return ;
}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_ncopy_sve_v4_pack4_core_ref
(
	const INDEXINT remainK,//列数
    const INDEXINT remainM,//行数
    const DEF_DOUBLE_AB* A,
    const INDEXINT lda,
    DEF_DOUBLE_C* Alocal
)
{
    const INDEXINT SSBLAS_K_PACK_SIZE=4;
    const INDEXINT FJBLAS_INNER_M_SIZE=COMPUTE_SIMDROW_NUM_IN_GEMM;
    const INDEXINT SIMD_I32_NUM=ssblasGemmBatchedEx_SIMDLENGTH<INDEXINT,DEF_DOUBLE_AB>()/4;
    const INDEXINT FJBLAS_BASE_MSIMD=SIMD_I32_NUM*FJBLAS_INNER_M_SIZE;
	//printf("SIMD_I32_NUM:%d\n",SIMD_I32_NUM);
	//printf("FJBLAS_BASE_MSIMD:%d\n",FJBLAS_BASE_MSIMD);

    const INDEXINT Alocal_COL=((remainK+SSBLAS_K_PACK_SIZE-1)/SSBLAS_K_PACK_SIZE)*FJBLAS_BASE_MSIMD;

    for(INDEXINT ik=0;ik<remainK;ik+=SSBLAS_K_PACK_SIZE){
        //remainMがFJBLAS_BASE_MSIMD以上あるとき
        INDEXINT im=0;
        for(im=0;im+FJBLAS_BASE_MSIMD<=remainM;im+=FJBLAS_BASE_MSIMD){
            ssblaschar4 tmp;
            for(INDEXINT i=0;i<FJBLAS_BASE_MSIMD;i++){
                tmp.x=A[(im+i)+(ik+0)*lda];
                tmp.y=ik+1<remainK?A[(im+i)+(ik+1)*lda]:0;
                tmp.z=ik+2<remainK?A[(im+i)+(ik+2)*lda]:0;
                tmp.w=ik+3<remainK?A[(im+i)+(ik+3)*lda]:0;
                Alocal[i+(ik/SSBLAS_K_PACK_SIZE)*FJBLAS_BASE_MSIMD+(im/FJBLAS_BASE_MSIMD)*Alocal_COL]=tmp;
            }
        }
        //printf("remainK: %d\n",remainK);
        //printf("im: %d\n",im);
        //端数
        for(;im<remainM;im+=FJBLAS_BASE_MSIMD){
            //printf("remainM: %d\n",remainM);
            //printf("remainK: %d\n",remainK);
            //printf("im: %d\n",im);
            ssblaschar4 tmp;
            for(INDEXINT i=0;im+i<remainM;i++){
                tmp.x=A[(im+i)+(ik+0)*lda];
                tmp.y=ik+1<remainK?A[(im+i)+(ik+1)*lda]:0;
                tmp.z=ik+2<remainK?A[(im+i)+(ik+2)*lda]:0;
                tmp.w=ik+3<remainK?A[(im+i)+(ik+3)*lda]:0;
                Alocal[i+(ik/SSBLAS_K_PACK_SIZE)*(remainM-im)+(im/FJBLAS_BASE_MSIMD)*Alocal_COL]=tmp;
                //printf("%d %d %d %d  %d\n",(int)tmp.x,(int)tmp.y,(int)tmp.z,(int)tmp.w, (im/FJBLAS_BASE_MSIMD)*Alocal_COL);
            }
        }
    }
}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_ncopy_sve_v4_pack4_core
(
	const INDEXINT remainK,//列数
    const INDEXINT remainM,//行数
    const DEF_DOUBLE_AB* A,
    const INDEXINT lda,
    DEF_DOUBLE_AB* Alocal_
)
{
	//printf("Xgemm_ncopy_sve_v4_pack4_core\n");
	DEF_DOUBLE_C* Alocal=(DEF_DOUBLE_C*)Alocal_;
    const INDEXINT SSBLAS_K_PACK_SIZE=sizeof(DEF_DOUBLE_C)/sizeof(DEF_DOUBLE_AB);
    const INDEXINT FJBLAS_INNER_M_SIZE=COMPUTE_SIMDROW_NUM_IN_GEMM;
    const INDEXINT SIMD_I32_NUM=ssblasGemmBatchedEx_SIMDLENGTH<INDEXINT,DEF_DOUBLE_AB>()/SSBLAS_K_PACK_SIZE;
    const INDEXINT FJBLAS_BASE_MSIMD=SIMD_I32_NUM*FJBLAS_INNER_M_SIZE;
	//printf("SIMD_I32_NUM:%d\n",SIMD_I32_NUM);
	//printf("FJBLAS_BASE_MSIMD:%d\n",FJBLAS_BASE_MSIMD);

    const INDEXINT Alocal_COL=((remainK+SSBLAS_K_PACK_SIZE-1)/SSBLAS_K_PACK_SIZE)*FJBLAS_BASE_MSIMD;

	INDEXINT ik=0;
    for(ik=0;ik+SSBLAS_K_PACK_SIZE<=remainK;ik+=SSBLAS_K_PACK_SIZE){
        //remainMがFJBLAS_BASE_MSIMD以上あるとき
        INDEXINT im=0;
        for(im=0;im+FJBLAS_BASE_MSIMD<=remainM;im+=FJBLAS_BASE_MSIMD){
            /*ssblaschar4 tmp;
            for(INDEXINT i=0;i<FJBLAS_BASE_MSIMD;i++){
                tmp.x=A[(im+i)+(ik+0)*lda];
                tmp.y=ik+1<remainK?A[(im+i)+(ik+1)*lda]:0;
                tmp.z=ik+2<remainK?A[(im+i)+(ik+2)*lda]:0;
                tmp.w=ik+3<remainK?A[(im+i)+(ik+3)*lda]:0;
                Alocal[i+(ik/SSBLAS_K_PACK_SIZE)*FJBLAS_BASE_MSIMD+(im/FJBLAS_BASE_MSIMD)*Alocal_COL]=tmp;
            }*/

			// using namespace ssblasEx::cpu::vecdef;
			// auto v0=vec_ld1(&A[(im)+(ik+0)*lda]);
			// auto v1=vec_ld1(&A[(im)+(ik+1)*lda]);
			// auto v2=vec_ld1(&A[(im)+(ik+2)*lda]);
			// auto v3=vec_ld1(&A[(im)+(ik+3)*lda]);
			// vec_st4(
			// 	(DEF_DOUBLE_AB*)(&Alocal[(ik/SSBLAS_K_PACK_SIZE)*FJBLAS_BASE_MSIMD+(im/FJBLAS_BASE_MSIMD)*Alocal_COL]),
			// 	v0,v1,v2,v3
			// );

			using namespace ssblasEx::cpu::vecdef;
			vec_col_pack4<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
			(
				&A[(im)+(ik+0)*lda], lda,
				&Alocal[(ik/SSBLAS_K_PACK_SIZE)*FJBLAS_BASE_MSIMD+(im/FJBLAS_BASE_MSIMD)*Alocal_COL]
			);
        }
		
		//im+=FJBLAS_BASE_MSIMD;
        //端数
        for(;im<remainM;im+=FJBLAS_BASE_MSIMD){
            ssblaschar4 tmp;
            for(INDEXINT i=0;im+i<remainM;i++){
                tmp.x=A[(im+i)+(ik+0)*lda];
                tmp.y=ik+1<remainK?A[(im+i)+(ik+1)*lda]:0;
                tmp.z=ik+2<remainK?A[(im+i)+(ik+2)*lda]:0;
                tmp.w=ik+3<remainK?A[(im+i)+(ik+3)*lda]:0;
                Alocal[i+(ik/SSBLAS_K_PACK_SIZE)*(remainM-im)+(im/FJBLAS_BASE_MSIMD)*Alocal_COL]=tmp;
				//printf("%d %d %d %d  %d\n",(int)tmp.x,(int)tmp.y,(int)tmp.z,(int)tmp.w, (im/FJBLAS_BASE_MSIMD)*Alocal_COL);
            }
        }
    }
	//return ;
    for(;ik<remainK;ik+=SSBLAS_K_PACK_SIZE){
        //remainMがFJBLAS_BASE_MSIMD以上あるとき
        INDEXINT im=0;
        for(im=0;im+FJBLAS_BASE_MSIMD<=remainM;im+=FJBLAS_BASE_MSIMD){
            ssblaschar4 tmp;
            for(INDEXINT i=0;i<FJBLAS_BASE_MSIMD;i++){
                tmp.x=A[(im+i)+(ik+0)*lda];
                tmp.y=ik+1<remainK?A[(im+i)+(ik+1)*lda]:0;
                tmp.z=ik+2<remainK?A[(im+i)+(ik+2)*lda]:0;
                tmp.w=ik+3<remainK?A[(im+i)+(ik+3)*lda]:0;
                Alocal[i+(ik/SSBLAS_K_PACK_SIZE)*FJBLAS_BASE_MSIMD+(im/FJBLAS_BASE_MSIMD)*Alocal_COL]=tmp;
            }
        }
        //端数
        for(;im<remainM;im+=FJBLAS_BASE_MSIMD){
            ssblaschar4 tmp;
            for(INDEXINT i=0;im+i<remainM;i++){
                tmp.x=A[(im+i)+(ik+0)*lda];
                tmp.y=ik+1<remainK?A[(im+i)+(ik+1)*lda]:0;
                tmp.z=ik+2<remainK?A[(im+i)+(ik+2)*lda]:0;
                tmp.w=ik+3<remainK?A[(im+i)+(ik+3)*lda]:0;
                Alocal[i+(ik/SSBLAS_K_PACK_SIZE)*(remainM-im)+(im/FJBLAS_BASE_MSIMD)*Alocal_COL]=tmp;
            }
        }
    }
}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_ncopy_sve_v4_pack4
(
	const INDEXINT k,//列数
    const INDEXINT m,//行数
    const DEF_DOUBLE_AB* A,
    const INDEXINT lda,
    DEF_DOUBLE_AB* Alocal,
	const INDEXINT TEAM_SIZE, const INDEXINT TEAM_ID
)
{
	/*if(TEAM_ID==0){
		Xgemm_ncopy_sve_v4_pack4_core
		<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
		(
			k, m,
			A, lda,
			Alocal
		);
	}
	return ;*/
	const INDEXINT SSBLAS_K_PACK_SIZE=sizeof(DEF_DOUBLE_C)/sizeof(DEF_DOUBLE_AB);
	const INDEXINT BASE_M_SIZE=4*ssblasGemmBatchedEx_SIMDLENGTH<INDEXINT,DEF_DOUBLE_AB>()/SSBLAS_K_PACK_SIZE;
	const INDEXINT BASE_K_SIZE=64;

	const INDEXINT BASE_K_BLOCK_SIZE=BASE_M_SIZE*BASE_K_SIZE;
	const INDEXINT BASE_M_BLOCK_SIZE=(BASE_M_SIZE)* ((k+SSBLAS_K_PACK_SIZE-1)/SSBLAS_K_PACK_SIZE) *SSBLAS_K_PACK_SIZE;//DEF_DOUBLE_AB基準

	INDEXINT JOB_M_NUM=(m+BASE_M_SIZE-1)/BASE_M_SIZE;
	INDEXINT JOB_K_NUM=(k+BASE_K_SIZE-1)/BASE_K_SIZE;

	//printf("JOB_M_NUM: %d\n",JOB_M_NUM);
	//printf("JOB_K_NUM: %d\n",JOB_K_NUM);
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
		Xgemm_ncopy_sve_v4_pack4_core
		<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
		(
			remain_k, remain_m,
			&A[START_M_INDEX+START_K_INDEX*lda], lda,
			&Alocal[
				ldlocal*(START_K_INDEX)
				+M_JOB_INDEX*BASE_M_BLOCK_SIZE]
		);
	}
	
}


// template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
// void Xgemm_tcopy_sve_v4_selector(INDEXINT k,INDEXINT m,const DEF_DOUBLE_AB *a,INDEXINT lda,DEF_DOUBLE_AB *aw)
// {
// 	//printf("Xgemm_tcopy_sve_v4_selector\n");
// 	if constexpr (std::is_same_v<DEF_DOUBLE_AB, float>) Xgemm_tcopy_sve_v4<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>(k, m, a, lda, aw);
// 	//else if constexpr (std::is_same_v<DEF_DOUBLE_AB, double>) Xgemm_tcopy_sve_v4<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>(k, m, a, lda, aw);
// 	//else if constexpr (std::is_same_v<DEF_DOUBLE_AB, int>) Xgemm_tcopy_sve_v4<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>(k, m, a, lda, aw);
// 	else if constexpr (std::is_same_v<DEF_DOUBLE_AB, char>) Xgemm_tcopy_sve_v4_pack4<INDEXINT, T_SCALE, DEF_DOUBLE_AB, ssblaschar4>(k, m, a, lda, (ssblaschar4*)aw);

// 	//printf("Xgemm_tcopy_sve_v4_selector done\n");
// }

}
}
}
}
}
#endif