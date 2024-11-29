#ifndef SSBLAS_LAUNCH_LAUNCHBLOCKSIZE_T
#define SSBLAS_LAUNCH_LAUNCHBLOCKSIZE_T

namespace ssblasEx{
namespace cpu{
namespace GemmBatchedEx{
namespace impl{
namespace simd{

template<typename INDEXINT>
struct LAUNCHBLOCKSIZE_t{
    INDEXINT BLOCK_M;//ブロッキングサイズ
    INDEXINT BLOCK_N;
    INDEXINT BLOCK_K;

    INDEXINT ASSIGN_M;//グループが担当する行列積のサイズ
    INDEXINT ASSIGN_N;
    INDEXINT ASSIGN_K;

    INDEXINT ASSIGN_Mtasks;//行列積のタスク数
    INDEXINT ASSIGN_Ntasks;
    INDEXINT ASSIGN_Ktasks;
    INDEXINT ASSIGN_batchtasks;
    INDEXINT ASSIGN_ALLTASKS;

    INDEXINT THREAD_PER_GROUP;

    size_t sharedmemory_size;
    size_t localmemory_size;

    bool ACOPY;
    bool BCOPY;
};

}
}
}
}
}


#endif