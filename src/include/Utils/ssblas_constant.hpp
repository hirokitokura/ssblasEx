#ifndef SSBLAS_CONSTANT
#define SSBLAS_CONSTANT

namespace ssblasEx{
namespace cpu{
namespace utils{
//CPU 構成
constexpr int EXPECT_CMG_THREAD_NUM=12;
constexpr int EXPECT_L1_WAY_NUM=4;
constexpr int EXPECT_L1_WAY_SIZE=16*1024;
constexpr int EXPECT_L1_TOTAL_SIZE=EXPECT_L1_WAY_NUM*EXPECT_L1_WAY_SIZE;
constexpr size_t EXPECT_L1_BAD_LDB_SIZE=EXPECT_L1_WAY_SIZE/2;




//行列積の際に一度に行う最大の列数
//ldbやldcが2048の倍数の場合においてスラッシングを低減する。
//THREAD_PER_COLの倍数でないが気にしない。
constexpr int COMPUTE_SIMDROW_NUM_IN_GEMM=4;
constexpr int COMPUTE_COL_NUM_IN_GEMM=5;
constexpr int THREAD_PER_COL=128;

constexpr int BCOPY_LOCAL_LDB=256;
constexpr int MAX_COMPUTE_C_COL_NUM=1024;

constexpr int MAX_THREADTEAM_NUM=256;
//const int EXPECT_THREADNUM_IN_THREADTEAM[5]={2,3,4,6,12};
const int EXPECT_THREADNUM_IN_THREADTEAM[1]={12};


constexpr size_t ALIGN_SIZE=256;

}
}
}
#endif