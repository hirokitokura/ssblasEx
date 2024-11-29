#ifndef SSBLAS_OPTMNPARA_T
#define SSBLAS_OPTMNPARA_T

namespace ssblasEx{
namespace cpu{
namespace utils{

template<typename INDEXINT>
struct OPTMNPARA_t{
    INDEXINT target_threadnum;
    INDEXINT m_para;
    INDEXINT n_para;
    INDEXINT k_para;
    INDEXINT batch_para;
    INDEXINT COMPUTE_C_COL_NUM;
    INDEXINT COMPUTE_C_ROW_NUM;
    INDEXINT COMPUTE_A_K_NUM;
    INDEXINT TOTAL_BLOCK_NUM;
};

}
}
}

#endif