#ifndef __DEF_GENERIC_ACLE
#define __DEF_GENERIC_ACLE
namespace ssblasEx{
namespace cpu{
namespace vecdef{


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void vec_col_pack4
(
    const DEF_DOUBLE_AB* A, const INDEXINT lda,
    DEF_DOUBLE_C* dest
)
{
    //if constexpr ((std::is_same_v<DEF_DOUBLE_AB, SSBLAS_SCHAR>)&&(std::is_same_v<DEF_DOUBLE_C, int>)){
    if constexpr ((sizeof(DEF_DOUBLE_AB)==1)&&(sizeof(DEF_DOUBLE_C)==4)){
        PTRUE_1B(p0);
        LD1B_ZXI(z0,p0,&A[lda*0],0);
        LD1B_ZXI(z1,p0,&A[lda*1],0);
        LD1B_ZXI(z2,p0,&A[lda*2],0);
        LD1B_ZXI(z3,p0,&A[lda*3],0);

        ZIP1_1B(z4, z0, z2);
        ZIP2_1B(z5, z0, z2);
        ZIP1_1B(z6, z1, z3);
        ZIP2_1B(z7, z1, z3);

        ZIP1_1B(z0, z4, z6);
        ZIP2_1B(z1, z4, z6);
        ZIP1_1B(z2, z5, z7);
        ZIP2_1B(z3, z5, z7);

        ST4B_ZXI(z0, p0, dest, 0);
        ST4B_ZXI(z1, p0, dest, 1);
        ST4B_ZXI(z2, p0, dest, 2);
        ST4B_ZXI(z3, p0, dest, 3);
    }
}




}
}
}

#endif