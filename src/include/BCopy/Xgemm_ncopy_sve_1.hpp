/***************************************************************************
(c) RIKEN 2024, 2024. All rights reserved. sgemm_ncopy_sve_1.c 0.3.26
Copyright 2024 FUJITSU limited
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

#ifndef XGEMM_NCOPY_SVE_1
#define XGEMM_NCOPY_SVE_1
//#include "def_sve_asm.h"
//#include <arm_sve.h>
namespace ssblasEx{
namespace cpu{
namespace xgemm{
namespace simd{
namespace Bcopy{
  
using namespace ssblasEx::cpu::utils;

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_ncopy_sve_1(INDEXINT m, INDEXINT n, const DEF_DOUBLE_AB * a, INDEXINT lda, DEF_DOUBLE_AB *b, INDEXINT ldbl){
  INDEXINT im, in;
  const DEF_DOUBLE_AB *a_offset, *a_offset_p;
  DEF_DOUBLE_AB *b_offset, *b_offset_p;
  const INDEXINT nvl=ssblasGemmBatchedEx_SIMDLENGTH<INDEXINT,DEF_DOUBLE_AB>();
  const INDEXINT nul=nvl*4;

  a_offset = a;
  a_offset_p = a+lda*2;
  b_offset = b;
  b_offset_p = b+m*2;

  PTRUE_PS(p0);
  for(in=0;in<n;in++) {
    for(im=0;im<m-nul+1;im+=nul) {
      LD1W_ZXI(z0,p0,a_offset,0);
      LD1W_ZXI(z1,p0,a_offset,1);
      LD1W_ZXI(z2,p0,a_offset,2);
      LD1W_ZXI(z3,p0,a_offset,3);

      ST1W_ZXI(z0,p0,b_offset,0);
      ST1W_ZXI(z1,p0,b_offset,1);
      ST1W_ZXI(z2,p0,b_offset,2);
      ST1W_ZXI(z3,p0,b_offset,3);
      PRFM_XI(PLDL2KEEP,a_offset_p,0);
      PRFM_XI(PSTL2KEEP,b_offset_p,0);
      a_offset+=nul;
      b_offset+=nul;
      a_offset_p+=nul;
      b_offset_p+=nul;
    }
    if(m-im>=nvl) {
      for(;im<m-nvl+1;im+=nvl) {
        LD1W_ZXI(z0,p0,a_offset,0);
        ST1W_ZXI(z0,p0,b_offset,0);
        a_offset+=nvl;
        b_offset+=nvl;
        a_offset_p+=nvl;
        b_offset_p+=nvl;
      }
    }
    if(m-im>0) {
      WHILELT_PSX(p1,im,m);
      LD1W_ZXI(z0,p1,a_offset,0);
      ST1W_ZXI(z0,p1,b_offset,0);
      a_offset+=m-im;
      b_offset+=m-im;
      a_offset_p+=m-im;
      b_offset_p+=m-im;
      //PRFM_XI(PLDL2KEEP,a_offset_p,-1);
      //PRFM_XI(PSTL2KEEP,b_offset_p,-1);
    }
    a_offset+=lda-m; 
    a_offset_p+=lda-m; 
    b_offset+=ldbl-m;
    b_offset_p+=ldbl-m;
  }


}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_ncopy_sve_v1_pack4(
    const INDEXINT remainK,
    const INDEXINT remainN,
    const DEF_DOUBLE_AB* B,
    const INDEXINT ldb,
    DEF_DOUBLE_C* Blocal,
    const INDEXINT ldbl
)
{
  constexpr INDEXINT PACK_SIZE=sizeof(DEF_DOUBLE_C)/sizeof(DEF_DOUBLE_AB);
  const INDEXINT nvl=ssblasGemmBatchedEx_SIMDLENGTH<INDEXINT,DEF_DOUBLE_AB>();
  //printf("nvl: %d\n",(int)nvl);
  if constexpr (std::is_same_v<DEF_DOUBLE_AB, SSBLAS_SCHAR>||std::is_same_v<DEF_DOUBLE_AB, SSBLAS_UCHAR>) {
    PTRUE_1B(p0);
    for(INDEXINT j=0;j<remainN;j++){
      const DEF_DOUBLE_AB* Bpad=&B[j*ldb];
      DEF_DOUBLE_AB* tmp=(DEF_DOUBLE_AB*)&Blocal[0];
      DEF_DOUBLE_AB* Blocalpad=&tmp[j*(ldbl*PACK_SIZE)];
      INDEXINT i=0;
      INDEXINT loop_num=0;
      loop_num=(remainK-i)/(8*nvl);
      //8回転
      for(INDEXINT loop=0;loop<loop_num;loop++){
        LD1B_ZXI(z0, p0, &Bpad[i], 0);
        LD1B_ZXI(z1, p0, &Bpad[i], 1);
        LD1B_ZXI(z2, p0, &Bpad[i], 2);
        LD1B_ZXI(z3, p0, &Bpad[i], 3);
        LD1B_ZXI(z4, p0, &Bpad[i], 4);
        LD1B_ZXI(z5, p0, &Bpad[i], 5);
        LD1B_ZXI(z6, p0, &Bpad[i], 6);
        LD1B_ZXI(z7, p0, &Bpad[i], 7);
        ST1B_ZXI(z0, p0, &Blocalpad[i], 0);
        ST1B_ZXI(z1, p0, &Blocalpad[i], 1);
        ST1B_ZXI(z2, p0, &Blocalpad[i], 2);
        ST1B_ZXI(z3, p0, &Blocalpad[i], 3);
        ST1B_ZXI(z4, p0, &Blocalpad[i], 4);
        ST1B_ZXI(z5, p0, &Blocalpad[i], 5);
        ST1B_ZXI(z6, p0, &Blocalpad[i], 6);
        ST1B_ZXI(z7, p0, &Blocalpad[i], 7);
        i+=8*nvl;
      }
      loop_num=(remainK-i)/(4*nvl);
      //4回転
      for(INDEXINT loop=0;loop<loop_num;loop++){
        LD1B_ZXI(z0, p0, &Bpad[i], 0);
        LD1B_ZXI(z1, p0, &Bpad[i], 1);
        LD1B_ZXI(z2, p0, &Bpad[i], 2);
        LD1B_ZXI(z3, p0, &Bpad[i], 3);
        ST1B_ZXI(z0, p0, &Blocalpad[i], 0);
        ST1B_ZXI(z1, p0, &Blocalpad[i], 1);
        ST1B_ZXI(z2, p0, &Blocalpad[i], 2);
        ST1B_ZXI(z3, p0, &Blocalpad[i], 3);
        i+=4*nvl;
      }
      loop_num=(remainK-i)/(2*nvl);
      //2回転
      for(INDEXINT loop=0;loop<loop_num;loop++){
        LD1B_ZXI(z0, p0, (&Bpad[i]), 0);
        LD1B_ZXI(z1, p0, (&Bpad[i]), 1);
        ST1B_ZXI(z0, p0, (&Blocalpad[i]), 0);
        ST1B_ZXI(z1, p0, (&Blocalpad[i]), 1);
        i+=(2*nvl);
      }
      loop_num=(remainK-i)/(1*nvl);
      //1回転
      for(INDEXINT loop=0;loop<loop_num;loop++){
        LD1B_ZXI(z0, p0, (&Bpad[i]), 0);
        ST1B_ZXI(z0, p0, (&Blocalpad[i]), 0);
        i+=(1*nvl);
      }
      //端数
      for(;i<remainK;i+=4){
        Blocalpad[i+0]=i+0<remainK?Bpad[i+0]:0;
        Blocalpad[i+1]=i+1<remainK?Bpad[i+1]:0;
        Blocalpad[i+2]=i+2<remainK?Bpad[i+2]:0;
        Blocalpad[i+3]=i+3<remainK?Bpad[i+3]:0;
      }
    }
  }
  else{
    for(INDEXINT j=0;j<remainN;j++){
      for(INDEXINT i=0;i<remainK;i+=4){
      ssblaschar4 tmp;
      tmp.x=B[(i+0)+j*ldb];
      tmp.y=i+1<remainK?B[(i+1)+j*ldb]:0;
      tmp.z=i+2<remainK?B[(i+2)+j*ldb]:0;
      tmp.w=i+3<remainK?B[(i+3)+j*ldb]:0;

      Blocal[i/4+j*ldbl]=tmp;
      }
    }
  }
}

//template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
//void Xgemm_ncopy_sve_1_selector(INDEXINT m, INDEXINT n, const DEF_DOUBLE_AB * a, INDEXINT lda, DEF_DOUBLE_AB *b, INDEXINT ldbl){
  ////Xgemm_ncopy_sve_1<INDEXINT, T_SCALE, float, float>(m, n, a, lda, b);
  //if constexpr (std::is_same_v<DEF_DOUBLE_AB, float>) Xgemm_ncopy_sve_1<INDEXINT, T_SCALE, float, float>(m, n, a, lda, b);
  ////else if constexpr (std::is_same_v<DEF_DOUBLE_AB, double>) Xgemm_ncopy_sve_1<INDEXINT, T_SCALE, double, double>(m, n, a, lda, b);
  //else if constexpr (std::is_same_v<DEF_DOUBLE_AB, char>) Xgemm_tcopy_sve_v1_pack4<INDEXINT, T_SCALE, DEF_DOUBLE_AB, ssblaschar4>(m, n, a, lda, (ssblaschar4*)b, ldbl);
//}

}
}
}
}
}
#endif