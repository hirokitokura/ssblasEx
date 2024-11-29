#ifndef SSBLAS_ISFUNCS
#define SSBLAS_ISFUNCS

namespace ssblasEx{
namespace cpu{
namespace utils{

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
bool isBcopy(
    const INDEXINT k_remain, 
    ssblasOperation_t transb,
    const INDEXINT ldb
)
{
    constexpr INDEXINT PACK_SIZE=sizeof(DEF_DOUBLE_C)/sizeof(DEF_DOUBLE_AB);
    bool retval=false;


    if(
        (k_remain%PACK_SIZE!=0)
        ||((PACK_SIZE!=1)&&(ldb%PACK_SIZE!=0))
        ||(SSBLAS_OP_T==transb)
        ||((sizeof(DEF_DOUBLE_C)*ldb)%EXPECT_L1_BAD_LDB_SIZE)==0)
    {
        retval=true;    
    }

    return retval;
}

}
}
}

#endif