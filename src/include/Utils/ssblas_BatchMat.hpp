#ifndef BATCHMAT_CLASS
#define BATCHMAT_CLASS

namespace ssblasEx{
namespace cpu{
namespace utils{

template<typename INDEXINT, typename DEF_DOUBLE>
class BatchMat {
    DEF_DOUBLE* const* MAT;
    INDEXINT BATCH_INDEX;
    INDEXINT ROW_INDEX;
    INDEXINT COL_INDEX;
    INDEXINT LDX;

    ssblasOperation_t transX;

public:
    BatchMat(ssblasOperation_t transX, INDEXINT batch_index, INDEXINT row_index, INDEXINT col_index, DEF_DOUBLE* const* mat, INDEXINT ldx)
    : transX(transX), BATCH_INDEX(batch_index), ROW_INDEX(row_index), COL_INDEX(col_index), MAT(mat), LDX(ldx) {}

    BatchMat(INDEXINT row_index, INDEXINT col_index, DEF_DOUBLE* const* mat, INDEXINT ldx)
    : BatchMat(SSBLAS_OP_N, 0, row_index, col_index, mat, ldx) {}

    BatchMat(ssblasOperation_t transX, DEF_DOUBLE* const* mat, INDEXINT ldx)
    : BatchMat(transX, 0, 0, 0, mat, ldx) {}

    BatchMat(DEF_DOUBLE* const* mat, INDEXINT ldx)
    : BatchMat(SSBLAS_OP_N, 0, 0, 0, mat, ldx) {}

    DEF_DOUBLE* ptr(INDEXINT batch_index, INDEXINT row_index, INDEXINT col_index) const {
        if(SSBLAS_OP_N==transX){
            return &MAT[BATCH_INDEX + batch_index][(ROW_INDEX + row_index) + (COL_INDEX + col_index) * LDX];
        }
        else if(SSBLAS_OP_T==transX){
            return &MAT[BATCH_INDEX + batch_index][(ROW_INDEX + row_index) * LDX + (COL_INDEX + col_index) ];
        }
        return NULL;//エラー
    }

    DEF_DOUBLE* ptr() const {
        return ptr(0, 0, 0);
    }

    INDEXINT ldx() const {
        return this->LDX;
    }

    ssblasOperation_t transx() const {
        return this->transX;
    }
};

}
}
}

#endif