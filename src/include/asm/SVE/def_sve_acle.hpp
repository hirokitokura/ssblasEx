#ifndef __DEF_SVE_ACLE
#define __DEF_SVE_ACLE
namespace ssblasEx{
namespace cpu{
namespace vecdef{

#if 0
#include<arm_sve.h>


static auto vec_pture_all()
{
    return svptrue_b8(); 
}

template<typename DEF_DOUBLE_AB>
constexpr auto vec_pture()
{
    if constexpr(sizeof(DEF_DOUBLE_AB)==1){
        return svptrue_b8();
    }
    else if constexpr(sizeof(DEF_DOUBLE_AB)==2){
        return svptrue_b16();
    }
    else if constexpr(sizeof(DEF_DOUBLE_AB)==4){
        return svptrue_b32();
    }
    else if constexpr(sizeof(DEF_DOUBLE_AB)==8){
        return svptrue_b64();
    }
    else {
        return -1;
    }
}
template<typename DEF_DOUBLE_AB>
auto vec_ld1(const DEF_DOUBLE_AB* A)
{
    return svld1(vec_pture<DEF_DOUBLE_AB>(), A);
}


template<typename DEF_VEC_DOUBLE>
auto make_vec4(DEF_VEC_DOUBLE v0, DEF_VEC_DOUBLE v1, DEF_VEC_DOUBLE v2, DEF_VEC_DOUBLE v3)
{
    return svcreate4(v0, v1, v2, v3);
}
template<typename DEF_DOUBLE_AB,typename DEF_VEC_DOUBLE>
void vec_st4(
    DEF_DOUBLE_AB* A,
    DEF_VEC_DOUBLE v0, DEF_VEC_DOUBLE v1, DEF_VEC_DOUBLE v2, DEF_VEC_DOUBLE v3)
{
    //referene
    //svst4(vec_pture_all(), A, svcreate4(v0, v1, v2, v3));

    DEF_VEC_DOUBLE t0, t1, t2, t3;
    t0=svzip1(v0, v2);
    t1=svzip2(v0, v2);
    t2=svzip1(v1, v3);
    t3=svzip2(v1, v3);

    v0=svzip1(t0, t2);
    v1=svzip2(t0, t2);
    v2=svzip1(t1, t3);
    v3=svzip2(t1, t3);

    svst1_vnum(vec_pture_all(), A, 0, v0);
    svst1_vnum(vec_pture_all(), A, 1, v1);
    svst1_vnum(vec_pture_all(), A, 2, v2);
    svst1_vnum(vec_pture_all(), A, 3, v3);
}

#endif

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void vec_col_pack4
(
    const DEF_DOUBLE_AB* A, const INDEXINT lda,
    DEF_DOUBLE_C* dest
)
{
    //if constexpr ((std::is_same_v<DEF_DOUBLE_AB, SSBLAS_SCHAR>)&&(std::is_same_v<DEF_DOUBLE_C, int>)){
    if constexpr ((sizeof(DEF_DOUBLE_AB)==1)&&(sizeof(DEF_DOUBLE_C)==4)){
        //PTRUE_1B(p0);
        //LD1B_ZXI(z0,p0,&A[lda*0],0);
        //LD1B_ZXI(z1,p0,&A[lda*1],0);
        //LD1B_ZXI(z2,p0,&A[lda*2],0);
        //LD1B_ZXI(z3,p0,&A[lda*3],0);
        LDR(z0, &A[lda*0],0);
        LDR(z1, &A[lda*1],0);
        LDR(z2, &A[lda*2],0);
        LDR(z3, &A[lda*3],0);

        ZIP1_1B(z4, z0, z2);
        ZIP2_1B(z5, z0, z2);
        ZIP1_1B(z6, z1, z3);
        ZIP2_1B(z7, z1, z3);

        ZIP1_1B(z0, z4, z6);
        ZIP2_1B(z1, z4, z6);
        ZIP1_1B(z2, z5, z7);
        ZIP2_1B(z3, z5, z7);

        // ST4B_ZXI(z0, p0, dest, 0);
        // ST4B_ZXI(z1, p0, dest, 1);
        // ST4B_ZXI(z2, p0, dest, 2);
        // ST4B_ZXI(z3, p0, dest, 3);

        STR(z0, dest, 0);
        STR(z1, dest, 1);
        STR(z2, dest, 2);
        STR(z3, dest, 3);
    }
}




}
}
}

#endif