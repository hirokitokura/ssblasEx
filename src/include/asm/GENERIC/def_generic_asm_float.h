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

#ifndef __DEF_GENERIC_ASM_FLOAT
#define __DEF_GENERIC_ASM_FLOAT

template<typename DEF_DOUBLE_C>
void genericfmla(regemu& z3, maskemu& p0, regemu& z1, regemu& z2){
  DEF_DOUBLE_C* z3p = (DEF_DOUBLE_C*)z3.get();
  DEF_DOUBLE_C* z1p = (DEF_DOUBLE_C*)z1.get();
  DEF_DOUBLE_C* z2p = (DEF_DOUBLE_C*)z2.get();
  int* p0p = (int*)p0.get();
  for(int i=0;i<__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__; i+=sizeof(DEF_DOUBLE_C)){
    if(p0p[i]==1){
      z3p[i/sizeof(DEF_DOUBLE_C)]=z3p[i/sizeof(DEF_DOUBLE_C)]+z1p[i/sizeof(DEF_DOUBLE_C)]*z2p[i/sizeof(DEF_DOUBLE_C)];
    }else{

    }
  }
}

template<typename DEF_DOUBLE_AB,typename DEF_DOUBLE_C>
void genericfmla_dot(regemu& z3, maskemu& p0, regemu& z1, regemu& z2){
  DEF_DOUBLE_C* z3p = (DEF_DOUBLE_C*)z3.get();
  DEF_DOUBLE_AB* z1p = (DEF_DOUBLE_AB*)z1.get();
  DEF_DOUBLE_AB* z2p = (DEF_DOUBLE_AB*)z2.get();
  int* p0p = (int*)p0.get();
  for(int i=0;i<__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__; i+=sizeof(DEF_DOUBLE_C)){
    if(p0p[i]==1){
      DEF_DOUBLE_C tmp=0;
      for(int j=0;j<sizeof(DEF_DOUBLE_C)/sizeof(DEF_DOUBLE_AB);j++){
        tmp+=(DEF_DOUBLE_C)(z1p[(i/sizeof(DEF_DOUBLE_AB))+j])*(DEF_DOUBLE_C)(z2p[(i/sizeof(DEF_DOUBLE_AB))+j]);
      }
      z3p[i/sizeof(DEF_DOUBLE_C)]=z3p[i/sizeof(DEF_DOUBLE_C)]+tmp;
    }else{

    }
  }
}


/* fmla(predicated) zds3=zds3+zs1*zs2 p/M */
#define FMLA_ZSP(zds3,pg,zs1,zs2) FMLA_ZSP_base(zds3,pg,zs1,zs2)
#define FMLA_ZSP_base(zds3,pg,zs1,zs2) \
{\
  if constexpr ((std::is_same_v<DEF_DOUBLE_AB, float>&&std::is_same_v<DEF_DOUBLE_C, float>)) \
    genericfmla<DEF_DOUBLE_C>(zds3, pg, zs1, zs2);\
  else if constexpr ((std::is_same_v<DEF_DOUBLE_AB, double>&&std::is_same_v<DEF_DOUBLE_C, double>)) \
    genericfmla<DEF_DOUBLE_C>(zds3, pg, zs1, zs2);\
  else if constexpr ((std::is_same_v<DEF_DOUBLE_AB, char>&&std::is_same_v<DEF_DOUBLE_C, int>)) \
    genericfmla_dot<DEF_DOUBLE_AB,DEF_DOUBLE_C>(zds3, pg, zs1, zs2);\
} 


//zds1=zs3+zds1*zs2  p/M
template<typename DEF_DOUBLE_C>
void genericfmad(regemu& z1, maskemu& p0, regemu& z2, regemu& z3){
  DEF_DOUBLE_C* z3p = (DEF_DOUBLE_C*)z3.get();
  DEF_DOUBLE_C* z1p = (DEF_DOUBLE_C*)z1.get();
  DEF_DOUBLE_C* z2p = (DEF_DOUBLE_C*)z2.get();
  int* p0p = (int*)p0.get();
  for(int i=0;i<__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__; i+=sizeof(DEF_DOUBLE_C)){
    if(p0p[i]==1){
      z1p[i/sizeof(DEF_DOUBLE_C)]=z3p[i/sizeof(DEF_DOUBLE_C)]+z1p[i/sizeof(DEF_DOUBLE_C)]*z2p[i/sizeof(DEF_DOUBLE_C)];
    }else{

    }
  }
}
/* fmad(predicated) zds1=zs3+zds1*zs2  p/M */
#define FMAD_ZSP(zds1,pg,zs2,zs3) FMAD_ZSP_base(zds1,pg,zs2,zs3)
#define FMAD_ZSP_base(zds1,pg,zs2,zs3) \
{\
  genericfmad<DEF_DOUBLE_C>(zds1,pg,zs2,zs3);\
}


/* fmul(predicated) zds1*=zs2  p/M */
template<typename DEF_DOUBLE_C>
void genericfmul(regemu& z1, maskemu& p0, regemu& z2){
  DEF_DOUBLE_C* z1p = (DEF_DOUBLE_C*)z1.get();
  DEF_DOUBLE_C* z2p = (DEF_DOUBLE_C*)z2.get();
  int* p0p = (int*)p0.get();
  for(int i=0;i<__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__; i+=sizeof(DEF_DOUBLE_C)){
    if(p0p[i]==1){
      z1p[i/sizeof(DEF_DOUBLE_C)]=z1p[i/sizeof(DEF_DOUBLE_C)]*z2p[i/sizeof(DEF_DOUBLE_C)];
    }else{

    }
  }
}
/* fmul(predicated) zds1*=zs2  p/M */
#define FMUL_ZSP(zds1,pg,zs2) FMUL_ZSP_base(zds1,pg,zs2)
#define FMUL_ZSP_base(zds1,pg,zs2) \
{\
  genericfmul<DEF_DOUBLE_C>(zds1,pg,zs2);\
}

void genericld1_1B(regemu& zt, maskemu& p0, const signed char* x1, int imm){
  signed char* ztp = (signed char*)zt.get();
  int* p0p = (int*)p0.get();
  for(int i=0;i<__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__; i+=sizeof(signed char)){
    if(p0p[i]==1){
      ztp[i/sizeof(signed char)]=x1[i/sizeof(signed char)+imm*svcnt<signed char>()];
    }else{

    }
  }
}

#define LD1B_ZXI(zt,pg,x1,imm) LD1B_ZXI_base(zt,pg,x1,imm)
#define LD1B_ZXI_base(zt,pg,x1,imm) \
{\
  genericld1_1B(zt,pg,x1,imm);\
}

template<typename DEF_DOUBLE_C>
void genericld1(regemu& zt, maskemu& p0, const DEF_DOUBLE_C* x1, int imm){
  DEF_DOUBLE_C* ztp = (DEF_DOUBLE_C*)zt.get();
  int* p0p = (int*)p0.get();
  for(int i=0;i<__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__; i+=sizeof(DEF_DOUBLE_C)){
    if(p0p[i]==1){
      ztp[i/sizeof(DEF_DOUBLE_C)]=x1[i/sizeof(DEF_DOUBLE_C)+imm*svcnt<DEF_DOUBLE_C>()];
    }else{

    }
  }
}
#define LD1W_ZXI(zt,pg,x1,imm) LD1W_ZXI_base(zt,pg,x1,imm)
#define LD1W_ZXI_base(zt,pg,x1,imm) \
{\
  genericld1(zt,pg,x1,imm);\
}


template<typename DEF_DOUBLE_C>
void genericld1r(regemu& zt, maskemu& p0, const DEF_DOUBLE_C* x1, int imm){
  DEF_DOUBLE_C* ztp = (DEF_DOUBLE_C*)zt.get();
  int* p0p = (int*)p0.get();
  DEF_DOUBLE_C tmp;
  tmp=x1[imm/sizeof(DEF_DOUBLE_C)];
  //std::cout << "genericld1r genericld1r" << std::endl;
  for(int i=0;i<__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__; i+=sizeof(DEF_DOUBLE_C)){
    if(p0p[i]==1){
      ztp[i/sizeof(DEF_DOUBLE_C)]=tmp;
    }else{

    }
    //std::cout << ztp[i/sizeof(DEF_DOUBLE_C)] << std::endl;
  }
  //std::cout << "genericld1r genericld1r end" << std::endl;
}
#define LD1RW_ZXI(zt,pg,x1,imm) LD1RW_ZXI_base(zt,pg,x1,imm)
#define LD1RW_ZXI_base(zt,pg,x1,imm) \
{ \
    genericld1r(zt,pg,x1,imm); \
}



void genericst1_4B(regemu& zt, maskemu& p0, int* x1, int imm){
  int* ztp = (int*)zt.get();
  int* p0p = (int*)p0.get();
  for(int i=0;i<__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__; i+=sizeof(int)){
    if(p0p[i]==1){
      x1[i/sizeof(int)+imm*svcnt<int>()]=ztp[i/sizeof(int)];
    }else{

    }
  }
}

#define ST4B_ZXI(zt,pg,x1,imm) ST4B_ZXI_base(zt,pg,x1,imm)
#define ST4B_ZXI_base(zt,pg,x1,imm) \
{\
  genericst1_4B(zt,pg,(int*)x1,imm);\
}


template<typename DEF_DOUBLE_C>
void genericst1(regemu& zt, maskemu& p0, DEF_DOUBLE_C* x1, int imm){
  DEF_DOUBLE_C* ztp = (DEF_DOUBLE_C*)zt.get();
  int* p0p = (int*)p0.get();
  for(int i=0;i<__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__; i+=sizeof(DEF_DOUBLE_C)){
    if(p0p[i]==1){
      x1[i/sizeof(DEF_DOUBLE_C)+imm*svcnt<DEF_DOUBLE_C>()]=ztp[i/sizeof(DEF_DOUBLE_C)];
    }else{

    }
  }
}
#define ST1W_ZXI(zt,pg,x1,imm) ST1W_ZXI_base(zt,pg,x1,imm)
#define ST1W_ZXI_base(zt,pg,x1,imm) \
{\
  genericst1(zt,pg,x1,imm);\
}


void genericzip1_1B(regemu& ZD, regemu& ZN, regemu& ZM){
  signed char* zd=(signed char*)ZD.get();
  signed char* zn=(signed char*)ZN.get();
  signed char* zm=(signed char*)ZM.get();
  for(int i=0;i<__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__; i+=sizeof(signed char)*2){
    
      zd[(i+0)/sizeof(signed char)]=zn[i/sizeof(signed char)];
      zd[(i+sizeof(signed char))/sizeof(signed char)]=zn[i/sizeof(signed char)];
  }
}
#define ZIP1_1B(ZD, ZN, ZM) \
{\
  genericzip1_1B(ZD, ZN, ZM); \
}

void genericzip2_1B(regemu& ZD, regemu& ZN, regemu& ZM){
  signed char* zd=(signed char*)ZD.get();
  signed char* zn=(signed char*)ZN.get();
  signed char* zm=(signed char*)ZM.get();
  for(int i=0;i<__SSBLAS_GENERIC_SIMD_BITS/__CHAR_BIT__; i+=sizeof(signed char)*2){
    
      zd[(i+0)/sizeof(signed char)]=zn[(i+__SSBLAS_GENERIC_SIMD_BITS/2)/sizeof(signed char)];
      zd[(i+sizeof(signed char))/sizeof(signed char)]=zn[(i+__SSBLAS_GENERIC_SIMD_BITS/2)/sizeof(signed char)];
  }
}
#define ZIP2_1B(ZD, ZN, ZM) \
{\
  genericzip2_1B(ZD, ZN, ZM); \
}



#endif  /* __DEF_GENERIC_ASM_FLOAT */



