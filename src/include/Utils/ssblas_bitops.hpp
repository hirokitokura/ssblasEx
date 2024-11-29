#ifndef SSBLAS_BITOPS
#define SSBLAS_BITOPS


namespace ssblasEx{
namespace cpu{
namespace utils{

//highestOneBit
template<typename INDEXINT>
INDEXINT highestOneBit(INDEXINT input)
{
    if(input<0){
        return -1;
    }
    INDEXINT retval;
    if constexpr(sizeof(INDEXINT)==4){
        retval= (sizeof(INDEXINT)*__CHAR_BIT__) - (__builtin_clz(input)+1);
    }else if constexpr(sizeof(INDEXINT)==8){
        retval= (sizeof(INDEXINT)*__CHAR_BIT__) - (__builtin_clzl(input)+1);
    }
    
    retval= 1<<retval;
    return retval;
}


}
}
}

#endif