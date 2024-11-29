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

#ifndef __DEF_SVE_ASM_FLOAT
#define __DEF_SVE_ASM_FLOAT

/* fmla(predicated) zds3=zds3+zs1*zs2 p/M */
#define FMLA_ZSP(zds3,pg,zs1,zs2) FMLA_ZSP_base(zds3,pg,zs1,zs2)
#define FMLA_ZSP_base(zds3,pg,zs1,zs2) \
{\
  if constexpr ((std::is_same_v<DEF_DOUBLE_AB, float>&&std::is_same_v<DEF_DOUBLE_C, float>)) \
    __asm__ __volatile__("\t\t\tfmla "#zds3".s,"#pg"/M,"#zs1".s,"#zs2".s":::#zs1,#zs2,#zds3,#pg);\
  else if constexpr ((std::is_same_v<DEF_DOUBLE_AB, double>&&std::is_same_v<DEF_DOUBLE_C, double>)) \
    __asm__ __volatile__("\t\t\tfmla "#zds3".d,"#pg"/M,"#zs1".d,"#zs2".d":::#zs1,#zs2,#zds3,#pg);\
  else if constexpr ((std::is_same_v<DEF_DOUBLE_AB, SSBLAS_SCHAR>&&std::is_same_v<DEF_DOUBLE_C, int>)) \
    __asm__ __volatile__("\t\t\tsdot "#zds3".s, "#zs1".b,"#zs2".b":::#zs1,#zs2,#zds3);\
} 

/* fmad(predicated) zds1=zs3+zds1*zs2  p/M */
#define FMAD_ZSP(zds1,pg,zs2,zs3) FMAD_ZSP_base(zds1,pg,zs2,zs3)
#define FMAD_ZSP_base(zds1,pg,zs2,zs3) \
{\
  if constexpr ((std::is_same_v<DEF_DOUBLE_AB, float>&&std::is_same_v<DEF_DOUBLE_C, float>)) \
	  __asm__ __volatile__("\t\t\tfmad "#zds1".s,"#pg"/M,"#zs2".s,"#zs3".s":::#zds1,#zs2,#zs3,#pg);\
  else if constexpr ((std::is_same_v<DEF_DOUBLE_AB, double>&&std::is_same_v<DEF_DOUBLE_C, double>)) \
   __asm__ __volatile__("\t\t\tfmad "#zds1".d,"#pg"/M,"#zs2".d,"#zs3".d":::#zds1,#zs2,#zs3,#pg);\
  else if constexpr ((std::is_same_v<DEF_DOUBLE_AB, SSBLAS_SCHAR>&&std::is_same_v<DEF_DOUBLE_C, int>)) \
   __asm__ __volatile__("\t\t\tmad "#zds1".s,"#pg"/M,"#zs2".s,"#zs3".s":::#zds1,#zs2,#zs3,#pg);\
}

/* fmul(predicated) zds1*=zs2  p/M */
#define FMUL_ZSP(zds1,pg,zs2) FMUL_ZSP_base(zds1,pg,zs2)
#define FMUL_ZSP_base(zds1,pg,zs2) \
{\
if constexpr ((std::is_same_v<DEF_DOUBLE_AB, float>&&std::is_same_v<DEF_DOUBLE_C, float>)) \
	  __asm__ __volatile__("\t\t\tfmul "#zds1".s,"#pg"/M,"#zds1".s,"#zs2".s":::#zds1,#zs2,#pg);\
  else if constexpr ((std::is_same_v<DEF_DOUBLE_AB, double>&&std::is_same_v<DEF_DOUBLE_C, double>)) \
   __asm__ __volatile__("\t\t\tfmul "#zds1".d,"#pg"/M,"#zds1".d,"#zs2".d":::#zds1,#zs2,#pg);\
  else if constexpr ((std::is_same_v<DEF_DOUBLE_AB, SSBLAS_SCHAR>&&std::is_same_v<DEF_DOUBLE_C, int>)) \
   __asm__ __volatile__("\t\t\tmul "#zds1".s,"#pg"/M,"#zds1".s,"#zs2".s":::#zds1,#zs2,#pg);\
}
  //__asm__ __volatile__("\t\t\tfmul "#zds1".s,"#pg"/M,"#zds1".s,"#zs2".s":::#zds1,#zs2,#pg);

#define LDR(zt,x1,imm) \
  __asm__ __volatile__("\t\t\tLDR  "#zt", [%0,#"#imm",MUL VL]"::"r"(x1):#zt);

#define STR(zt,x1,imm) \
  __asm__ __volatile__("\t\t\tSTR  "#zt", [%0,#"#imm",MUL VL]"::"r"(x1):#zt);

#define LD1B_ZXI(zt,pg,x1,imm) LD1B_ZXI_base(zt,pg,x1,imm)
#define LD1B_ZXI_base(zt,pg,x1,imm) \
  __asm__ __volatile__("\t\t\tld1B {"#zt".b},"#pg"/Z,[%0,#"#imm",MUL VL]"::"r"(x1):#zt,#pg);

#define LD1W_ZXI(zt,pg,x1,imm) LD1W_ZXI_base(zt,pg,x1,imm)
#define LD1W_ZXI_base(zt,pg,x1,imm) \
{\
  if constexpr ((std::is_same_v<DEF_DOUBLE_AB, float>&&std::is_same_v<DEF_DOUBLE_C, float>)) \
    __asm__ __volatile__("\t\t\tld1w {"#zt".s},"#pg"/Z,[%0,#"#imm",MUL VL]"::"r"(x1):#zt,#pg);\
  else if constexpr ((std::is_same_v<DEF_DOUBLE_AB, double>&&std::is_same_v<DEF_DOUBLE_C, double>)) \
    __asm__ __volatile__("\t\t\tld1d {"#zt".d},"#pg"/Z,[%0,#"#imm",MUL VL]"::"r"(x1):#zt,#pg);\
  else if constexpr ((std::is_same_v<DEF_DOUBLE_AB, SSBLAS_SCHAR>&&std::is_same_v<DEF_DOUBLE_C, int>)) \
    __asm__ __volatile__("\t\t\tld1w {"#zt".s},"#pg"/Z,[%0,#"#imm",MUL VL]"::"r"(x1):#zt,#pg);\
}

#define LD1RW_ZXI(zt,pg,x1,imm) LD1RW_ZXI_base(zt,pg,x1,imm)
#define LD1RW_ZXI_base(zt,pg,x1,imm) \
{ \
    if constexpr (sizeof(DEF_DOUBLE_C)==4){\
      if constexpr (imm==0)\
        __asm__ __volatile__("\t\t\tld1rw "#zt".s,"#pg"/Z,[%0,#0]"::"r"(x1):#zt,#pg); \
      else if constexpr (imm==4)\
        __asm__ __volatile__("\t\t\tld1rw "#zt".s,"#pg"/Z,[%0,#4]"::"r"(x1):#zt,#pg); \
      else if constexpr (imm==8)\
        __asm__ __volatile__("\t\t\tld1rw "#zt".s,"#pg"/Z,[%0,#8]"::"r"(x1):#zt,#pg); \
      else if constexpr (imm==16)\
        __asm__ __volatile__("\t\t\tld1rw "#zt".s,"#pg"/Z,[%0,#16]"::"r"(x1):#zt,#pg); \
    }\
    else if constexpr (sizeof(DEF_DOUBLE_C)==8){\
      if constexpr (imm==0)\
        __asm__ __volatile__("\t\t\tld1rd "#zt".d,"#pg"/Z,[%0,#0]"::"r"(x1):#zt,#pg); \
      else if constexpr (imm==4)\
        __asm__ __volatile__("\t\t\tld1rd "#zt".d,"#pg"/Z,[%0,#4]"::"r"(x1):#zt,#pg); \
      else if constexpr (imm==8)\
        __asm__ __volatile__("\t\t\tld1rd "#zt".d,"#pg"/Z,[%0,#8]"::"r"(x1):#zt,#pg); \
      else if constexpr (imm==16)\
        __asm__ __volatile__("\t\t\tld1rd "#zt".d,"#pg"/Z,[%0,#16]"::"r"(x1):#zt,#pg); \
    }\
}

#define ST1W_ZXI(zt,pg,x1,imm) ST1W_ZXI_base(zt,pg,x1,imm)
#define ST1W_ZXI_base(zt,pg,x1,imm) \
{\
  if constexpr ((std::is_same_v<DEF_DOUBLE_AB, float>&&std::is_same_v<DEF_DOUBLE_C, float>)) \
    __asm__ __volatile__("\t\t\tst1w {"#zt".s},"#pg",[%0,#"#imm",MUL VL]"::"r"(x1):"memory",#zt,#pg);\
  else if constexpr ((std::is_same_v<DEF_DOUBLE_AB, double>&&std::is_same_v<DEF_DOUBLE_C, double>)) \
    __asm__ __volatile__("\t\t\tst1d {"#zt".d},"#pg",[%0,#"#imm",MUL VL]"::"r"(x1):"memory",#zt,#pg);\
  else if constexpr ((std::is_same_v<DEF_DOUBLE_AB, SSBLAS_SCHAR>&&std::is_same_v<DEF_DOUBLE_C, int>)) \
    __asm__ __volatile__("\t\t\tst1w {"#zt".s},"#pg",[%0,#"#imm",MUL VL]"::"r"(x1):"memory",#zt,#pg);\
}

#define ST4B_ZXI(zt,pg,x1,imm) ST4B_ZXI_base(zt,pg,x1,imm)
#define ST4B_ZXI_base(zt,pg,x1,imm) \
__asm__ __volatile__("\t\t\tst1w {"#zt".s},"#pg",[%0,#"#imm",MUL VL]"::"r"(x1):"memory",#zt,#pg);

#define ST1B_ZXI(zt,pg,x1,imm) ST1B_ZXI_base(zt,pg,x1,imm)
#define ST1B_ZXI_base(zt,pg,x1,imm) \
__asm__ __volatile__("\t\t\tst1b {"#zt".b},"#pg",[%0,#"#imm",MUL VL]"::"r"(x1):"memory",#zt,#pg);


#define ZIP1_1B(ZD, ZN, ZM) \
  __asm__ __volatile__("\t\t\t ZIP1 "#ZD".b, "#ZN".b, "#ZM".b":::#ZD, #ZN,#ZM);

#define ZIP2_1B(ZD, ZN, ZM) \
  __asm__ __volatile__("\t\t\t ZIP2 "#ZD".b, "#ZN".b, "#ZM".b":::#ZD,#ZN,#ZM);

#endif  /* __DEF_SVE_ASM_FLOAT */



