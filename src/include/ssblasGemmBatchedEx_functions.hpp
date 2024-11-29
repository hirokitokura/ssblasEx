#ifndef SSBLASBATCH_FUNCTIONS_INCLUDE
#define SSBLASBATCH_FUNCTIONS_INCLUDE


#include "asm/def_asm.h"

#include "Utils/ssblas_Utils.hpp"

#include"Launcher/launcher.hpp"



//#include "ACopy/Xgemm_ACopy.hpp"
#include "ACopy/Xgemm_ACopy_selector.hpp"
#include "BCopy/Xgemm_BCopy_selector.hpp"
#include "CScale/Xgemm_CScale_selector.hpp"
#include "GemmCore/Xgemm_CoreKernel_selector.hpp"

//#include"Runner/ssblasGemmBatchedEx_Runner.hpp"
//#include"Runner/ssblasGemmBatchedEx_Runner_BMKN.hpp"



#endif 
