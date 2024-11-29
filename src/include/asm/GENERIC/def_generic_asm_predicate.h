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

#ifndef __DEF_GENERIC_ASM_PREDICATE
#define __DEF_GENERIC_ASM_PREDICATE

/* ptrue  all true (no pattern) */
#define PTRUE_1B(p) \
{\
   p.ptrue<signed char>();\
}

/* ptrue  all true (no pattern) */
#define PTRUE_PS(p) PTRUE_PS_base(p)
#define PTRUE_PS_base(p) \
{\
   p.ptrue<DEF_DOUBLE_C>();\
}



#define WHILELT_PSX(p,x1,x2) WHILELT_PSX_base(p,x1,x2)
#define WHILELT_PSX_base(p,x1,x2) \
{\
   p.whilelt<DEF_DOUBLE_C>(x1,x2);\
}


#endif  /* __DEF_GENERIC_ASM_PREDICATE */


