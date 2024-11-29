#include<iostream>
//#include<arm_sve.h>


namespace ssblasEx{
namespace cpu{
namespace utils{

template<typename INDEXINT, typename DEF_DOUBLE>
INDEXINT ssblasGemmBatchedEx_SIMDLENGTH()
{
    
    //printf("ssblasGemmBatchedEx_SIMDLENGTH\n");
    INDEXINT retval;

    if constexpr (sizeof(DEF_DOUBLE)==1){
        //retval = svcntb();
        CNT_1B(retval);
    }else if constexpr (sizeof(DEF_DOUBLE)==2){
        //retval = svcnth(); 
        CNT_2B(retval);
    }else if constexpr (sizeof(DEF_DOUBLE)==4){
        //retval = svcntw(); 
        CNT_4B(retval);
    }else if constexpr (sizeof(DEF_DOUBLE)==8){
        //retval = svcntd(); 
        CNT_8B(retval);
    }else
        retval=0;
    return retval;

}

template<typename INDEXINT, typename DEF_DOUBLE>
static int ssblas_SIMDLEN(){
    return ssblasGemmBatchedEx_SIMDLENGTH<INDEXINT, DEF_DOUBLE>();
}

}
}
}