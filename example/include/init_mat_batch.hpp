#ifndef SSBLAS_EXAMPLE_INIT_MAT_BATCH
#define SSBLAS_EXAMPLE_INIT_MAT_BATCH
#include<iostream>
#include<random>

template<typename INDEXINT>
void init_mat_batch(const INDEXINT ROW, const INDEXINT COL, float** A, const INDEXINT lda, const INDEXINT BATCH)
{
    static INDEXINT seed=10;
    static std::mt19937_64 engine(seed);
    seed++;
    std::uniform_real_distribution<float> dist1(0.0, 1.0);

    for(INDEXINT batch=0;batch<BATCH;batch++){
        for(INDEXINT j=0;j<COL;j++){
            for(INDEXINT i=0;i<ROW;i++){
                A[batch][i+j*lda]=dist1(engine);
                //A[batch][i+j*lda]=1;
                //A[batch][i+j*lda]=i+j*ROW;
                //printf("%d\n", (int)A[i+j*lda]);
                //printf("%d\n", A[i+j*lda]);
                //A[i+j*lda]=1;
            }
        }
    }
}
template<typename INDEXINT>
void init_mat_batch(const INDEXINT ROW, const INDEXINT COL, double** A, const INDEXINT lda, const INDEXINT BATCH)
{
    static INDEXINT seed=10;
    static std::mt19937_64 engine(seed);
    seed++;
    std::uniform_real_distribution<double> dist1(0.0, 1.0);

    for(INDEXINT batch=0;batch<BATCH;batch++){
        for(INDEXINT j=0;j<COL;j++){
            for(INDEXINT i=0;i<ROW;i++){
                A[batch][i+j*lda]=dist1(engine);
            }
        }
    }
}

template<typename INDEXINT>
void init_mat_batch(const INDEXINT ROW, const INDEXINT COL, char** A, const INDEXINT lda, const INDEXINT BATCH)
{
    static INDEXINT seed=10;
    static std::mt19937_64 engine(seed);
    seed++;
    std::uniform_int_distribution<int> dist1(1, 2);

    for(INDEXINT batch=0;batch<BATCH;batch++){
        for(INDEXINT j=0;j<COL;j++){
            for(INDEXINT i=0;i<ROW;i++){
                A[batch][i+j*lda]=dist1(engine);
                //A[batch][i+j*lda]=1;
                //A[batch][i+j*lda]=i+j*ROW;
                //printf("%d\n", (int)A[i+j*lda]);
                //printf("%d\n", A[i+j*lda]);
                //A[i+j*lda]=1;
            }
        }
    }
}

template<typename INDEXINT>
void init_mat_batch(const INDEXINT ROW, const INDEXINT COL, signed char** A, const INDEXINT lda, const INDEXINT BATCH)
{
    static INDEXINT seed=10;
    static std::mt19937_64 engine(seed);
    seed++;
    std::uniform_int_distribution<int> dist1(1, 32);

    for(INDEXINT batch=0;batch<BATCH;batch++){
        for(INDEXINT j=0;j<COL;j++){
            for(INDEXINT i=0;i<ROW;i++){
                A[batch][i+j*lda]=dist1(engine);
                //A[batch][i+j*lda]=1;
                //A[batch][i+j*lda]=i+j*ROW;
                //printf("%d\n", (int)A[i+j*lda]);
                //printf("%d\n", A[i+j*lda]);
                //A[i+j*lda]=1;
            }
        }
    }
}

template<typename INDEXINT>
void init_mat_batch(const INDEXINT ROW, const INDEXINT COL, int** A, const INDEXINT lda, const INDEXINT BATCH)
{
    static INDEXINT seed=10;
    static std::mt19937_64 engine(seed);
    seed++;
    std::uniform_int_distribution<int> dist1(-64, 64);

    for(INDEXINT batch=0;batch<BATCH;batch++){
        for(INDEXINT j=0;j<COL;j++){
            for(INDEXINT i=0;i<ROW;i++){
                A[batch][i+j*lda]=dist1(engine);
            }
        }
    }
}

#endif