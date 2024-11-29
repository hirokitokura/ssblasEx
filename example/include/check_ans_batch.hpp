#ifndef SSBLAS_EXAMPLE_CHECK_ANS_BATCH
#define SSBLAS_EXAMPLE_CHECK_ANS_BATCH

template<typename INDEXINT>
static void check_ans_batch(
    const INDEXINT ROW, const INDEXINT COL,
    const float* C_ORI[],
    const float* C_TUNE[],
    const INDEXINT ldc,
    const INDEXINT BATCH
)
{

    float MAX_ERROR=0;
    for(INDEXINT batch=0;batch<BATCH;batch++){
        for(INDEXINT j=0;j<COL;j++){
            for(INDEXINT i=0;i<ROW;i++){
                float tmp=0;
                tmp=C_ORI[batch][i+j*ldc]-C_TUNE[batch][i+j*ldc];
                if(C_ORI[batch][i+j*ldc]!=0.0){
                    tmp/=C_ORI[batch][i+j*ldc];
                }
                tmp=fabs(tmp);
                MAX_ERROR=tmp>MAX_ERROR?tmp:MAX_ERROR;
                //printf("%f %f\n", C_ORI[batch][i+j*ldc],C_TUNE[batch][i+j*ldc]);
            }
        }
    }
    printf("MAX ERROR %e\n", MAX_ERROR);
}
template<typename INDEXINT>
static void check_ans_batch(
    const INDEXINT ROW, const INDEXINT COL,
    const double* C_ORI[],
    const double* C_TUNE[],
    const INDEXINT ldc,
    const INDEXINT BATCH
)
{

    double MAX_ERROR=0;
    for(INDEXINT batch=0;batch<BATCH;batch++){
        for(INDEXINT j=0;j<COL;j++){
            for(INDEXINT i=0;i<ROW;i++){
                double tmp=0;
                tmp=C_ORI[batch][i+j*ldc]-C_TUNE[batch][i+j*ldc];
                if(C_ORI[batch][i+j*ldc]!=0.0){
                    tmp/=C_ORI[batch][i+j*ldc];
                }
                tmp=fabs(tmp);
                MAX_ERROR=tmp>MAX_ERROR?tmp:MAX_ERROR;
                //printf("%f %f\n", C_ORI[batch][i+j*ldc],C_TUNE[batch][i+j*ldc]);
            }
        }
    }
    printf("MAX ERROR %e\n", MAX_ERROR);
}
template<typename INDEXINT>
static void check_ans_batch(
    const INDEXINT ROW, const INDEXINT COL,
    const int* C_ORI[],
    const int* C_TUNE[],
    const INDEXINT ldc,
    const INDEXINT BATCH
)
{

    int MAX_ERROR=0;
    for(INDEXINT batch=0;batch<BATCH;batch++){
        for(INDEXINT j=0;j<COL;j++){
            for(INDEXINT i=0;i<ROW;i++){
                float tmp=0;
                tmp=C_ORI[batch][i+j*ldc]-C_TUNE[batch][i+j*ldc];
                tmp=abs(tmp);
                MAX_ERROR=tmp>MAX_ERROR?tmp:MAX_ERROR;
                //printf("%d %d\n", C_ORI[batch][i+j*ldc],C_TUNE[batch][i+j*ldc]);
            }
        }
    }
    printf("MAX ERROR %d\n", MAX_ERROR);
}

#endif
