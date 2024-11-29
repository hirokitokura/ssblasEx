#ifndef MNPARA_BNKM_CONST
#define MNPARA_BNKM_CONST

namespace ssblasEx{
namespace cpu{
namespace GemmBatchedEx{
namespace impl{
namespace simd{

namespace SSBLAS_MNPARA_BNKM {
    constexpr int BASE_BLOCK_SIZE_N=1500;
    constexpr int BASE_SIMD_NUM_M=8;
    constexpr size_t BASE_SUBMATRIX_A_SIZE_IN_BYTE=512*1024;
    constexpr int BASE_LDBCOPY_SIZE_K=512;
}

}
}
}
}
}

#endif