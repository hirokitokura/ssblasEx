#ifndef SSBLAS_THREADS
#define SSBLAS_THREADS
#ifdef _OPENMP
#include<omp.h>
#endif

namespace ssblasEx{
namespace cpu{
namespace utils{

static int ssblas_GetThreadNum()
{
    int THREAD_NUM=1;

    #ifdef _OPENMP
    #pragma omp parallel
    {
        THREAD_NUM=omp_get_num_threads();
    }
    #endif

    //printf("ssblas_GetThreadnum: %d\n", THREAD_NUM);
    return THREAD_NUM;
}

static bool ssblas_acceptable_threadnum_for_tuneroute(const int THREAD_NUM)
{
    bool retval=false;

    if(THREAD_NUM<=0){
        return false;
    }
    
    if(THREAD_NUM%12==0){
        return true;
    }
    if(THREAD_NUM%12==6){
        return true;
    }
    if(THREAD_NUM%12==4){
        return true;
    }
    if(THREAD_NUM%12==3){
        return true;
    }
    if(THREAD_NUM%12==2){
        return true;
    }

    retval=false;
    return retval;
}

template<typename INDEXINT>
class ssblasteam{
    INDEXINT thread_id;
    INDEXINT thread_num;


    INDEXINT x;
    INDEXINT y;
    INDEXINT z;
};

}
}
}
#endif 