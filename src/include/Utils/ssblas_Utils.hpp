#ifndef SSBLAS_UTILS
#define SSBLAS_UTILS


#include "ssblasGemmBatchedEx_SIMDLENGTH.hpp"
#include "ssblas_threads.hpp"
#include "ssblas_constant.hpp"
#include "ssblas_bitops.hpp"
#include "ssblas_malloc.hpp"
#include "ssblas_isfuncs.hpp"
#include "ssblas_BatchMat.hpp"
#include "ssblas_computemode.hpp"
#include "ssblas_optmnpara.hpp"

namespace ssblasEx{
namespace cpu{
namespace utils{

template<typename INDEXINT,typename DEF_DOUBLE_C>
void ssblas_show_matrix
(
    const INDEXINT ROW,
    const INDEXINT COL,
    const DEF_DOUBLE_C* C,
    const INDEXINT ldc
)
{
    for(int i=0;i<ROW;i++){
        for(int j=0;j<COL;j++){
            std::cout << C[i+j*ldc] << " ";
        }
        std::cout << std::endl;
    }
}

}
}
}
#endif