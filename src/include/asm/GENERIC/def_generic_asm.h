/***************************************************************************
Copyright 2024 RIKEN
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

#ifndef __DEF_GENERIC_ASM_INLINE
#define __DEF_GENERIC_ASM_INLINE
//#include <arm_sve.h>

constexpr int __SSBLAS_GENERIC_SIMD_BITS=512;

template<typename INDEXINT>
static void  CNT_1B(INDEXINT& x){
  x=(__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__)/1;
}
template<typename INDEXINT>
static void  CNT_2B(INDEXINT& x){
  x=(__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__)/2;
}
template<typename INDEXINT>
static void  CNT_4B(INDEXINT& x){
  x=(__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__)/4;
}
template<typename INDEXINT>
static void  CNT_8B(INDEXINT& x){
  x=(__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__)/8;
}


static int svcntb(){
  return (__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__)/1;
}
static int svcnth(){
  return (__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__)/2;
}
static int svcntw(){
  return (__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__)/4;
}
static int svcntd(){
  return (__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__)/8;
}

template<typename DEF_DOUBLE_C>
int svcnt(){
    int  retval;

    if constexpr (sizeof(DEF_DOUBLE_C)==1)
        retval = svcntb();
    else if constexpr (sizeof(DEF_DOUBLE_C)==2)
        retval = svcnth(); 
    else if constexpr (sizeof(DEF_DOUBLE_C)==4)
        retval = svcntw(); 
    else if constexpr (sizeof(DEF_DOUBLE_C)==8)
        retval = svcntd(); 
    else
        retval=0;
    return retval;
}

class regemu{
  double reg[(__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__)/sizeof(double)];
  public:
  template<typename DEF_DOUBLE_C>
  void set(DEF_DOUBLE_C val){
    constexpr int LOOP_NUM=(__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__)/sizeof(DEF_DOUBLE_C);
    DEF_DOUBLE_C* reg_p=(DEF_DOUBLE_C*)reg;
    //std::cout << "regemu set" << std::endl;
    for(int i=0;i<LOOP_NUM;i++){
      reg_p[i]=(DEF_DOUBLE_C)val;
      //std::cout << reg_p[i] << std::endl;
    }
    //std::cout << "regemu set end" << std::endl;
  }

  double* get(){
    return &reg[0];
  }
};

class maskemu{
  int pg [__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__];
  public:
  template<typename DEF_DOUBLE_C>
  void ptrue(){
    for(int i=0;i<__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__; i+=sizeof(DEF_DOUBLE_C)){
      pg[i]=1;
    }
  }

  template<typename DEF_DOUBLE_C>
  void whilelt(int op1, int op2){
    //std::cout << "maskemu whilelt" << std::endl;
    for(int i=0;i<__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__; i+=sizeof(DEF_DOUBLE_C)){
      if(op1+i/sizeof(DEF_DOUBLE_C)<op2){
        pg[i]=1;
      }else{
        pg[i]=0;
      }
      //std::cout << pg[i] << std::endl;
    }
    //std::cout << "maskemu whilelt end" << std::endl;
  }

  int* get(){
    return &pg[0];
  }
};




__thread regemu z0;
__thread regemu z1;
__thread regemu z2;
__thread regemu z3;
__thread regemu z4;
__thread regemu z5;
__thread regemu z6;
__thread regemu z7;
__thread regemu z8;
__thread regemu z9;
__thread regemu z10;
__thread regemu z11;
__thread regemu z12;
__thread regemu z13;
__thread regemu z14;
__thread regemu z15;
__thread regemu z16;
__thread regemu z17;
__thread regemu z18;
__thread regemu z19;
__thread regemu z20;
__thread regemu z21;
__thread regemu z22;
__thread regemu z23;
__thread regemu z24;
__thread regemu z25;
__thread regemu z26;
__thread regemu z27;
__thread regemu z28;
__thread regemu z29;
__thread regemu z30;
__thread regemu z31;
__thread regemu z32;
                   
__thread maskemu p0;
__thread maskemu p1;
__thread maskemu p2;
__thread maskemu p3;
__thread maskemu p4;
__thread maskemu p5;
__thread maskemu p6;
__thread maskemu p7;
__thread maskemu p8;
__thread maskemu p9;
__thread maskemu p10;
__thread maskemu p11;
__thread maskemu p12;
__thread maskemu p13;
__thread maskemu p14;
__thread maskemu p15;

#define PRFM_XI(prfop,x,i)  /**/

#define PRFM_XXLSL3(prfop,x,i)  /**/

/*NOT USED*/
#define CNTW_X(x)  /**/

#define NOP  /**/

#define DUP_ZSI(Z,X) DUP_ZSI_base(Z,X)
#define DUP_ZSI_base(Z,X) \
{\
  Z.set(X);\
}
#define PLDL1KEEP 0
#define PLDL1STRM 1
#define PLDL2KEEP 2
#define PLDL2STRM 3
#define PLDL3KEEP 4
#define PLDL3STRM 5
#define PSTL1KEEP 8
#define PSTL1STRM 9
#define PSTL2KEEP 10
#define PSTL2STRM 11
#define PSTL3KEEP 12
#define PSTL3STRM 13

#include "def_generic_asm_float.h"
#include "def_generic_asm_predicate.h"
#include "def_generic_acle.hpp"


#endif   /* __DEF_GENERIC_ASM_INLINE */

