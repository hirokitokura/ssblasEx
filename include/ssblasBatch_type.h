#ifndef SSBLASBATCH_TYPE
#define SSBLASBATCH_TYPE


typedef long int SSINT;
typedef long int SSLONG ;

typedef enum {
  SSBLAS_int_32,
  SSBLAS_int_64
} ssblasIntType_t;


typedef enum {
  SSBLAS_STATUS_SUCCESS = 0,
  //CUBLAS_STATUS_NOT_INITIALIZED = 1,
  SSBLAS_STATUS_ALLOC_FAILED,
  SSBLAS_STATUS_INVALID_VALUE,
  SSBLAS_STATUS_INTERNAL_ERROR,
  SSBLAS_STATUS_NOTIMPLEMENTED_ERROR,
  SSBLAS_STATUS_NOT_COMPUTED_ERROR
} ssblasStatus_t;

static void ssblasShowError(ssblasStatus_t retval)
{
  if(SSBLAS_STATUS_SUCCESS==retval){
    printf("SSBLAS_STATUS_SUCCESS\n");
  }
  else if(SSBLAS_STATUS_ALLOC_FAILED==retval){
    printf("SSBLAS_STATUS_ALLOC_FAILED\n");
  }
  else if(SSBLAS_STATUS_INVALID_VALUE==retval){
    printf("SSBLAS_STATUS_INVALID_VALUE\n");
  }
  else if(SSBLAS_STATUS_INTERNAL_ERROR==retval){
    printf("SSBLAS_STATUS_INTERNAL_ERROR\n");
  }
  else if(SSBLAS_STATUS_NOTIMPLEMENTED_ERROR==retval){
    printf("SSBLAS_STATUS_NOTIMPLEMENTED_ERROR\n");
  }
  else if(SSBLAS_STATUS_NOT_COMPUTED_ERROR==retval){
    printf("SSBLAS_STATUS_NOT_COMPUTED_ERROR\n");
  }
  else{
        printf("SSBLAS_UNKNOWN\n");
    }
}


typedef enum {
  SSBLAS_OP_N = 0,
  SSBLAS_OP_T = 1,
  SSBLAS_OP_C = 2,
} ssblasOperation_t;



typedef signed char SSBLAS_SCHAR;
typedef unsigned char SSBLAS_UCHAR;

typedef enum {
  SSBLAS_R_8I,// char
  SSBLAS_R_8U,// unsigned char
  SSBLAS_R_16F,
  SSBLAS_R_32I,
  SSBLAS_R_32F,
  SSBLAS_R_64F
} ssblasDataType_t;

typedef enum {
  SSBLAS_COMPUTE_DEFAULT_TYPE,
  SSBLAS_COMPUTE_32I,
  SSBLAS_COMPUTE_32F
}ssblasComputeType_t ;


typedef enum {
    SSBLAS_COMPUTE_DEFAULT
}ssblasGemmAlgo_t ;




typedef struct{
    char x;
    char y;
    char z;
    char w;
}ssblaschar4;






typedef enum {
  SSBLAS_STATUS_COMPUTED = 0,
  SSBLAS_STATUS_NOT_COMPUTED
} ssblasinternalComputeStatus_t;




#endif