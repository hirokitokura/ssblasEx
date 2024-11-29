#ifndef DEF_ASM_HPP
#define DEF_ASM_HPP


#ifdef __ARM_FEATURE_SVE
#include"./SVE/def_sve_asm.h"

#else
#include"./GENERIC/def_generic_asm.h"
#endif

#endif