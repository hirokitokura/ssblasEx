#ifndef XGEMM_ACOPY_CLASS
#define XGEMM_ACOPY_CLASS

#include"Xgemm_ACopy_selector.hpp"
namespace ssblasEx{
namespace cpu{
namespace xgemm{
namespace simd{
namespace Acopy{
    template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
    class ACopy{
        private:
        static constexpr INDEXINT PACK_SIZE=sizeof(DEF_DOUBLE_C)/sizeof(DEF_DOUBLE_AB);
        DEF_DOUBLE_C* Alocal=NULL;
        INDEXINT ALLOC_ROW=0;
        INDEXINT ALLOC_COL=0;

        public:
        ACopy() {            
        }

        void init(INDEXINT ROW, INDEXINT COL){
            ALLOC_ROW=ROW;
            ALLOC_COL=COL;
            Alocal=(DEF_DOUBLE_C*)ssblas_malloc(sizeof(DEF_DOUBLE_C)*ALLOC_ROW*ALLOC_COL);
        }

        DEF_DOUBLE_C* getPtr(){
            return Alocal;
        }

        void release(){
            ssblas_free(Alocal);
            Alocal=NULL;
        }

        ~ACopy(){
        }

    };
}
}
}
}
}


#endif
