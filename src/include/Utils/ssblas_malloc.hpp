#ifndef SSBLAS_MALLOC
#define SSBLAS_MALLOC

namespace ssblasEx{
namespace cpu{
namespace utils{

static void* ssblas_malloc(const size_t malloc_size)
{
    //printf("%s: %lld\n", __func__,malloc_size );
    void* retval=NULL;
    if(malloc_size==0){
        return NULL;
    }

    posix_memalign(&retval,ALIGN_SIZE,malloc_size);

    //printf("%s: %lld %p\n", __func__,malloc_size, retval);
    return retval;
}


static void ssblas_free(void* ptr)
{
    //printf("%s: %p\n", __func__, ptr);
    free(ptr);
}

}
}
}
#endif