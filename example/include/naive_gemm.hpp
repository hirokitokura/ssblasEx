#ifndef SSBLAS_EXAMPLE_NAIVE_GEMM
#define SSBLAS_EXAMPLE_NAIVE_GEMM
template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void naive_gemm_NN
(
    INDEXINT                  M,
    INDEXINT                  N,
    INDEXINT                  K,
    const T_SCALE          alpha,
    const DEF_DOUBLE_AB *const    A[],
    const INDEXINT            lda,
    const DEF_DOUBLE_AB *const    B[],
    const INDEXINT            ldb,
    const T_SCALE          beta,
    DEF_DOUBLE_C *const          C[],
    const INDEXINT            ldc,
    const INDEXINT            batchCount
)
{

    #pragma omp parallel for collapse(3)
    for(INDEXINT batch=0;batch < batchCount; batch++ ){
        for(INDEXINT j=0;j<N;j++){
            for(INDEXINT i=0;i<M;i++){
                DEF_DOUBLE_C tmp=0;
                for(INDEXINT k=0;k<K;k++){
                    tmp+=(DEF_DOUBLE_C)A[batch][i+k*lda]*(DEF_DOUBLE_C)B[batch][k+j*ldb];
                    //printf("%f %f\n", A[batch][i+k*lda], B[batch][k+j*ldb]);
                    //printf("%d %d\n", (int)A[batch][i+k*lda], (int)B[batch][k+j*ldb]);
                }
                if(beta!=0){
                    C[batch][i+j*ldc]=alpha*tmp+beta*C[batch][i+j*ldc];
                }else{
                    C[batch][i+j*ldc]=alpha*tmp;
                }
                //printf("%d\n", C[batch][i+j*ldc]);
            }
        }
    }
}

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void naive_gemm_NT
(
    INDEXINT                  M,
    INDEXINT                  N,
    INDEXINT                  K,
    const T_SCALE          alpha,
    const DEF_DOUBLE_AB *const    A[],
    const INDEXINT            lda,
    const DEF_DOUBLE_AB *const    B[],
    const INDEXINT            ldb,
    const T_SCALE          beta,
    DEF_DOUBLE_C *const          C[],
    const INDEXINT            ldc,
    const INDEXINT            batchCount
)
{

    #pragma omp parallel for collapse(3)
    for(INDEXINT batch=0;batch < batchCount; batch++ ){
        for(INDEXINT j=0;j<N;j++){
            for(INDEXINT i=0;i<M;i++){
                DEF_DOUBLE_C tmp=0;
                for(INDEXINT k=0;k<K;k++){
                    tmp+=(DEF_DOUBLE_C)A[batch][i+k*lda]*(DEF_DOUBLE_C)B[batch][k*ldb+j];
                    //printf("%f %f\n", A[batch][i+k*lda], B[batch][k+j*ldb]);
                }
                if(beta!=0){
                    C[batch][i+j*ldc]=alpha*tmp+beta*C[batch][i+j*ldc];
                }else{
                    C[batch][i+j*ldc]=alpha*tmp;
                }
                //printf("%f\n", C[batch][i+j*ldc]);
            }
        }
    }
}

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void naive_gemm_TN
(
    INDEXINT                  M,
    INDEXINT                  N,
    INDEXINT                  K,
    const T_SCALE          alpha,
    const DEF_DOUBLE_AB *const    A[],
    const INDEXINT            lda,
    const DEF_DOUBLE_AB *const    B[],
    const INDEXINT            ldb,
    const T_SCALE          beta,
    DEF_DOUBLE_C *const          C[],
    const INDEXINT            ldc,
    const INDEXINT            batchCount
)
{
    //printf("naive_gemm_TN\n");
    #pragma omp parallel for collapse(3)
    for(INDEXINT batch=0;batch < batchCount; batch++ ){
        for(INDEXINT j=0;j<N;j++){
            for(INDEXINT i=0;i<M;i++){
                DEF_DOUBLE_C tmp=0;
                for(INDEXINT k=0;k<K;k++){
                    tmp+=(DEF_DOUBLE_C)A[batch][i*lda+k]*(DEF_DOUBLE_C)B[batch][k+j*ldb];
                    //printf("%f %f\n", A[batch][i+k*lda], B[batch][k+j*ldb]);
                }
                if(beta!=0){
                    C[batch][i+j*ldc]=alpha*tmp+beta*C[batch][i+j*ldc];
                }else{
                    C[batch][i+j*ldc]=alpha*tmp;
                }
                //printf("%f\n", C[batch][i+j*ldc]);
            }
        }
    }
}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void naive_gemm_TT
(
    INDEXINT                  M,
    INDEXINT                  N,
    INDEXINT                  K,
    const T_SCALE          alpha,
    const DEF_DOUBLE_AB *const    A[],
    const INDEXINT            lda,
    const DEF_DOUBLE_AB *const    B[],
    const INDEXINT            ldb,
    const T_SCALE          beta,
    DEF_DOUBLE_C *const          C[],
    const INDEXINT            ldc,
    const INDEXINT            batchCount
)
{

    #pragma omp parallel for collapse(3)
    for(INDEXINT batch=0;batch < batchCount; batch++ ){
        for(INDEXINT j=0;j<N;j++){
            for(INDEXINT i=0;i<M;i++){
                DEF_DOUBLE_C tmp=0;
                for(INDEXINT k=0;k<K;k++){
                    tmp+=(DEF_DOUBLE_C)A[batch][i*lda+k]*(DEF_DOUBLE_C)B[batch][k*ldb+j];
                    //printf("%f %f\n", A[batch][i+k*lda], B[batch][k+j*ldb]);
                }
                if(beta!=0){
                    C[batch][i+j*ldc]=alpha*tmp+beta*C[batch][i+j*ldc];
                }else{
                    C[batch][i+j*ldc]=alpha*tmp;
                }
                //printf("%f\n", C[batch][i+j*ldc]);
            }
        }
    }
}



#include<cblas.h>

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void naive_gemm
(
    ssblasOperation_t    transa,
    ssblasOperation_t    transb,
    INDEXINT                  m,
    INDEXINT                  n,
    INDEXINT                  k,
    const T_SCALE          alpha,
    const DEF_DOUBLE_AB *const    A[],
    const INDEXINT            lda,
    const DEF_DOUBLE_AB *const    B[],
    const INDEXINT            ldb,
    const T_SCALE          beta,
    DEF_DOUBLE_C *const          C[],
    const INDEXINT            ldc,
    const INDEXINT            batchCount
)
{
    CBLAS_TRANSPOSE TRANSA;
    CBLAS_TRANSPOSE TRANSB;
    if(SSBLAS_OP_N==transa){
        TRANSA=CblasNoTrans;
    }
    else if(SSBLAS_OP_T==transa){
        TRANSA=CblasTrans;
    }
    if(SSBLAS_OP_N==transb){
        TRANSB=CblasNoTrans;
    }
    else if(SSBLAS_OP_T==transb){
        TRANSB=CblasTrans;
    }

    if constexpr (std::is_same_v<T_SCALE, float>&&std::is_same_v<DEF_DOUBLE_AB, float>&&std::is_same_v<DEF_DOUBLE_C, float>)
    {
        for(INDEXINT batch=0;batch<batchCount;batch++)
        cblas_sgemm
        (
            CblasColMajor,
            TRANSA,
            TRANSB,
            m,n,k,
            alpha,
            A[batch],lda,
            B[batch],ldb,
            beta,
            C[batch],ldc
        );
    }
    else if constexpr (std::is_same_v<T_SCALE, double>&&std::is_same_v<DEF_DOUBLE_AB, double>&&std::is_same_v<DEF_DOUBLE_C, double>)
    {
        for(INDEXINT batch=0;batch<batchCount;batch++)
        cblas_dgemm
        (
            CblasColMajor,
            TRANSA,
            TRANSB,
            m,n,k,
            alpha,
            A[batch],lda,
            B[batch],ldb,
            beta,
            C[batch],ldc
        );
    }
    else
    {
        //printf("naive_gemm\n");
        //fflush(stdout);
        if((SSBLAS_OP_N==transa)&&(SSBLAS_OP_N==transb)){
            naive_gemm_NN<INDEXINT, T_SCALE,DEF_DOUBLE_AB, DEF_DOUBLE_C>
            (
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batchCount
            );
            return ;
        }
        else if((SSBLAS_OP_N==transa)&&(SSBLAS_OP_T==transb)){
            naive_gemm_NT<INDEXINT, T_SCALE,DEF_DOUBLE_AB, DEF_DOUBLE_C>
            (
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batchCount
            );
            return ;
        }
        else if((SSBLAS_OP_T==transa)&&(SSBLAS_OP_N==transb)){
            naive_gemm_TN<INDEXINT, T_SCALE,DEF_DOUBLE_AB, DEF_DOUBLE_C>
            (
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batchCount
            );
            return ;
        }
        else if((SSBLAS_OP_T==transa)&&(SSBLAS_OP_T==transb)){
            naive_gemm_TT<INDEXINT, T_SCALE,DEF_DOUBLE_AB, DEF_DOUBLE_C>
            (
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batchCount
            );
            return ;
        }
        else{
            return ;
        }
    }
}


#endif