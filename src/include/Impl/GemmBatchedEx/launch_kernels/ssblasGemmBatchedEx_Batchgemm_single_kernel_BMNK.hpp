#ifndef SSBLAS_LAUNCH_KERNELS_SINGLE_BMKN
#define SSBLAS_LAUNCH_KERNELS_SINGLE_BMKN
#include"../../../ssblasGemmBatchedEx_functions.hpp"

#include"../LAUNCHBLOCKSIZE_t.hpp"

namespace ssblasEx{
namespace cpu{
namespace GemmBatchedEx{
namespace impl{
namespace simd{

namespace single_BMNK_kernel{

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_ACopy_main(
    INDEXINT COL,INDEXINT ROW,
    const ssblasOperation_t DEF_TRANS, 
    const DEF_DOUBLE_AB *a,INDEXINT lda,
    DEF_DOUBLE_AB *aw,
    const INDEXINT TEAM_SIZE, const INDEXINT TEAM_ID
)
{
    using namespace ssblasEx::cpu::xgemm::simd::Acopy;
    Xgemm_ACopy<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>(COL, ROW, DEF_TRANS, a, lda, aw, TEAM_SIZE, TEAM_ID);

}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_ACopy_main(
    INDEXINT COL,INDEXINT ROW,
    const ssblasOperation_t DEF_TRANS, 
    const DEF_DOUBLE_AB *a,INDEXINT lda,
    DEF_DOUBLE_AB *aw
)
{
    Xgemm_ACopy_main<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
    (COL, ROW, DEF_TRANS, a, lda, aw, 1, 0);
}


template<typename INDEXINT, typename T_SCALE,  typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_BCopy_main(INDEXINT m, INDEXINT n, const ssblasOperation_t DEF_TRANS, const DEF_DOUBLE_AB * a, INDEXINT lda, DEF_DOUBLE_AB *b, INDEXINT ldbl){
    using namespace ssblasEx::cpu::xgemm::simd::Bcopy;
    Xgemm_BCopy<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>(m, n, DEF_TRANS, a, lda, b, ldbl);
}

template<typename INDEXINT, typename T_SCALE,typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void Xgemm_CScale_main
(
    const INDEXINT ROW,
    const INDEXINT COL,
    const T_SCALE alpha,
    DEF_DOUBLE_C* C,
    const INDEXINT ldc,
    const INDEXINT TEAM_SIZE, const INDEXINT TEAM_ID
)
{
    using namespace ssblasEx::cpu::xgemm::simd::Cscale;
    for(INDEXINT col_index=TEAM_ID;col_index<COL;col_index+=TEAM_SIZE){
        Xgemm_CScale_selector
        <INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
        (
           ROW,
           1,
           alpha,
           &C[0+col_index*ldc],
           ldc
        );
    }
}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
void  Xgemm_CoreKernel_main
(
	ssblasEx::cpu::launcher::ssblasKernelHandle_t KernelHandle,
	ssblasOperation_t transa,//Not used
    ssblasOperation_t transb,
	INDEXINT M,
	INDEXINT N,
	INDEXINT K,
	T_SCALE alpha, 
	const DEF_DOUBLE_AB* Ap,
	const DEF_DOUBLE_AB* B,
	INDEXINT ldb,
	DEF_DOUBLE_C* C, 
	INDEXINT ldc 
)
{
    using namespace ssblasEx::cpu::xgemm::simd::GemmCore;
	constexpr INDEXINT PACK_SIZE=sizeof(DEF_DOUBLE_C)/sizeof(DEF_DOUBLE_AB);
	constexpr int MNB=COMPUTE_COL_NUM_IN_GEMM;

	const INDEXINT thread_id=KernelHandle.ssblasIndex->threadIdx;
	const INDEXINT groupSize=KernelHandle.ssblasIndex->groupSize;
	const INDEXINT LDBX=(K+PACK_SIZE-1)/PACK_SIZE;

	const INDEXINT k_remain_local=(K+PACK_SIZE-1)/PACK_SIZE;

	BatchMat<INDEXINT, const DEF_DOUBLE_AB> matB(transb, &B, ldb);
	BatchMat<INDEXINT, DEF_DOUBLE_C> matC(&C, ldc);
	for(INDEXINT n_index=MNB*(thread_id);n_index<N;n_index+=MNB*groupSize){
		INDEXINT n_remain=n_index+MNB<N?MNB:N-n_index;
		if(
			isBcopy<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
			(K, matB.transx(), matB.ldx())
		){
			DEF_DOUBLE_AB* Blocal;
			Blocal=(DEF_DOUBLE_AB*)KernelHandle.local_ptr;

			Xgemm_BCopy_main<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
			(
				K,
				n_remain, 
				matB.transx(),
				matB.ptr(0, 0, n_index),
				matB.ldx(),
				Blocal,
				LDBX
			);
			
			Xgemm_CoreKernel<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
			(
				M,
				n_remain,
				k_remain_local,
				alpha,
				(const DEF_DOUBLE_C*)Ap,
				(const DEF_DOUBLE_C*)Blocal,
				LDBX,
				matC.ptr(0, 0, n_index),
				matC.ldx()
			);           
		}
		else{
			Xgemm_CoreKernel<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
				(
					M,
					n_remain,
					k_remain_local,
					alpha,
					(const DEF_DOUBLE_C*)Ap,
					(const DEF_DOUBLE_C*)matB.ptr(0, 0, n_index),
					matB.ldx()/PACK_SIZE,
					matC.ptr(0, 0, n_index),
					matC.ldx()
				);          
		}
	}

}


template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
ssblasStatus_t ssblasGemmBatchedEx_Batchgemm_single_BMNK_kernel_body
(
    ssblasEx::cpu::launcher::ssblasKernelHandle_t KernelHandle,
    ssblasOperation_t                             transa,
    ssblasOperation_t                             transb,
    INDEXINT                                      M,
    INDEXINT                                      N,
    INDEXINT                                      K,
    const T_SCALE                                 alpha,
    const DEF_DOUBLE_AB*                          A,
    const INDEXINT                                lda,
    const DEF_DOUBLE_AB*                          B,
    const INDEXINT                                ldb,
    const T_SCALE                                 beta,
    DEF_DOUBLE_C*                                 C,
    const INDEXINT                                ldc,
    LAUNCHBLOCKSIZE_t<INDEXINT>                   gemmparam
)
{
    //printf("%s %d\n",__FILE__, __LINE__);
    static constexpr INDEXINT PACK_SIZE=sizeof(DEF_DOUBLE_C)/sizeof(DEF_DOUBLE_AB);

    ssblasEx::cpu::utils::BatchMat<INDEXINT, const DEF_DOUBLE_AB> matA(transa, &A, lda);
    ssblasEx::cpu::utils::BatchMat<INDEXINT, const DEF_DOUBLE_AB> matB(transb, &B, ldb);
    ssblasEx::cpu::utils::BatchMat<INDEXINT, DEF_DOUBLE_C> matC(&C, ldc);
    
    INDEXINT FAKE_M;
    INDEXINT FAKE_N;
    INDEXINT FAKE_K;

    if(0)
    printf("%d: %d %d %d\n",
    omp_get_thread_num(),
    gemmparam.BLOCK_M,
    gemmparam.BLOCK_N,
    gemmparam.BLOCK_K
    );
    DEF_DOUBLE_AB* Alocal=(DEF_DOUBLE_AB*)KernelHandle.shared_ptr;
    //#pragma omp barrier
    //Mのループ
    for(INDEXINT START_INDEX_M=0;START_INDEX_M<M;START_INDEX_M+=gemmparam.BLOCK_M){
        FAKE_M=START_INDEX_M+gemmparam.BLOCK_M<M?gemmparam.BLOCK_M:M-START_INDEX_M;

        for(INDEXINT START_INDEX_N=0;START_INDEX_N<N;START_INDEX_N+=gemmparam.BLOCK_N){
            //printf("%d %d,%d,%d (%d,%d)\n",omp_get_thread_num(), START_INDEX_M, START_INDEX_K, START_INDEX_N, START_INDEX_N, N);
            FAKE_N=START_INDEX_N+gemmparam.BLOCK_N<N?gemmparam.BLOCK_N:N-START_INDEX_N;
            
            //Kのループ
            for(INDEXINT START_INDEX_K=0;START_INDEX_K<K;START_INDEX_K+=gemmparam.BLOCK_K){
                FAKE_K=START_INDEX_K+gemmparam.BLOCK_K<K?gemmparam.BLOCK_K:K-START_INDEX_K;

                if(START_INDEX_K==0){
                    Xgemm_CScale_main
                    <INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
                    (
                        FAKE_M,
                        FAKE_N,
                        beta,
                        matC.ptr(0, START_INDEX_M, START_INDEX_N),
                        matC.ldx(),
                        KernelHandle.ssblasIndex->groupSize,KernelHandle.ssblasIndex->threadIdx
                    );
                    //printf("%s %d\n",__FILE__, __LINE__);
                    KernelHandle.barrier();
                }

                KernelHandle.barrier();
                //Aの読み込み
                Xgemm_ACopy_main<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
                (
                    FAKE_K, FAKE_M,
                    matA.transx(),
                    matA.ptr(0, START_INDEX_M, START_INDEX_K),
                    matA.ldx(),
                    (DEF_DOUBLE_AB*)Alocal,
                    KernelHandle.ssblasIndex->groupSize,KernelHandle.ssblasIndex->threadIdx
                );
                KernelHandle.barrier();

                //GEMM
                Xgemm_CoreKernel_main
                <INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
                (
                    KernelHandle,
                    transa,
                    transb,
                    FAKE_M,
                    FAKE_N,
                    FAKE_K,
                    alpha,
                    Alocal,
                    (matB.ptr(0, START_INDEX_K, START_INDEX_N)),
                    matB.ldx(),
                    matC.ptr(0, START_INDEX_M, START_INDEX_N),
                    matC.ldx()
                );
            }
            KernelHandle.barrier();
        }
        KernelHandle.barrier();
    }
    //printf("END %d  %s %d\n", omp_get_thread_num(), __FILE__, __LINE__);
    return SSBLAS_STATUS_SUCCESS;
}

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
ssblasStatus_t ssblasGemmBatchedEx_Batchgemm_single_BMNK_kernel
(
    ssblasEx::cpu::launcher::ssblasKernelHandle_t KernelHandle,
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
    DEF_DOUBLE_C *const         C[],
    const INDEXINT            ldc,
    const INDEXINT            batchCount,
    LAUNCHBLOCKSIZE_t<INDEXINT> gemmparam
)
{
    //KernelHandle.ssblasIndex;
    //KernelHandle.ssblasBarrier;
    //KernelHandle.shared_ptr;

    INDEXINT JOB_INDEX;
    JOB_INDEX=KernelHandle.ssblasIndex->groupIdx;

    INDEXINT JOBINDEX_M;
    INDEXINT JOBINDEX_N;
    INDEXINT JOBINDEX_BATCH;
    JOBINDEX_M=JOB_INDEX%gemmparam.ASSIGN_Mtasks;
    JOBINDEX_N=(JOB_INDEX/gemmparam.ASSIGN_Mtasks)%gemmparam.ASSIGN_Ntasks;
    JOBINDEX_BATCH=((JOB_INDEX/gemmparam.ASSIGN_Mtasks)/gemmparam.ASSIGN_Ntasks);

    INDEXINT STARI_INDEX_M;
    INDEXINT STARI_INDEX_N;
    INDEXINT STARI_INDEX_BATCH;
    STARI_INDEX_M=JOBINDEX_M*gemmparam.ASSIGN_M;
    STARI_INDEX_N=JOBINDEX_N*gemmparam.ASSIGN_N;
    STARI_INDEX_BATCH=JOBINDEX_BATCH;

    INDEXINT FAKE_M;
    INDEXINT FAKE_N;
    INDEXINT FAKE_K;

    FAKE_M=STARI_INDEX_M+gemmparam.ASSIGN_M<m?gemmparam.ASSIGN_M:m-STARI_INDEX_M;
    FAKE_N=STARI_INDEX_N+gemmparam.ASSIGN_N<n?gemmparam.ASSIGN_N:n-STARI_INDEX_N;
    FAKE_K=k;

    ssblasEx::cpu::utils::BatchMat<INDEXINT, const DEF_DOUBLE_AB> matA(transa, STARI_INDEX_BATCH, STARI_INDEX_M, 0, A, lda);
    ssblasEx::cpu::utils::BatchMat<INDEXINT, const DEF_DOUBLE_AB> matB(transb, STARI_INDEX_BATCH, 0, STARI_INDEX_N, B, ldb);
    ssblasEx::cpu::utils::BatchMat<INDEXINT, DEF_DOUBLE_C> matC(SSBLAS_OP_N, STARI_INDEX_BATCH, STARI_INDEX_M, STARI_INDEX_N, C, ldc);

    ssblasGemmBatchedEx_Batchgemm_single_BMNK_kernel_body
    <INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>
    (
        KernelHandle,
        transa,
        transb,
        FAKE_M,
        FAKE_N,
        FAKE_K,
        alpha,
        (const DEF_DOUBLE_AB*)matA.ptr(),
        matA.ldx(),
        (const DEF_DOUBLE_AB*)matB.ptr(),
        matB.ldx(),
        beta,
        (DEF_DOUBLE_C*)matC.ptr(),
        matC.ldx(),
        gemmparam
    );
    //printf("%s %d\n",__FILE__, __LINE__);
    return SSBLAS_STATUS_SUCCESS;
}

}

template<typename INDEXINT, typename T_SCALE, typename DEF_DOUBLE_AB, typename DEF_DOUBLE_C>
ssblasStatus_t ssblasGemmBatchedEx_Batchgemm_single_BMNK_host
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
    DEF_DOUBLE_C *const         C[],
    const INDEXINT            ldc,
    const INDEXINT            batchCount
)
{
    ssblasStatus_t retval=SSBLAS_STATUS_SUCCESS;

    using namespace ssblasEx::cpu::utils;
    const INDEXINT THREAD_NUM=ssblas_GetThreadNum();
    const INDEXINT CMG_NUM=THREAD_NUM/EXPECT_CMG_THREAD_NUM;
    const INDEXINT THREAD_PER_CMG=THREAD_NUM/CMG_NUM;
    LAUNCHBLOCKSIZE_t<INDEXINT> gemmparam;
    //ブロッキングサイズの決定
    {
        const INDEXINT SIMDLEN=ssblasEx::cpu::utils::ssblas_SIMDLEN<INDEXINT, DEF_DOUBLE_C>();
        const INDEXINT PREFER_M=SIMDLEN*ssblasEx::cpu::utils::COMPUTE_SIMDROW_NUM_IN_GEMM;

        constexpr INDEXINT PACK_SIZE=sizeof(DEF_DOUBLE_C)/sizeof(DEF_DOUBLE_AB);
	    constexpr int MNB=COMPUTE_COL_NUM_IN_GEMM;
        {
            gemmparam.THREAD_PER_GROUP=1;
            gemmparam.BLOCK_M=256;
            gemmparam.BLOCK_N=ssblasEx::cpu::utils::THREAD_PER_COL;
            gemmparam.BLOCK_K=512;

            gemmparam.ASSIGN_Mtasks=((m+gemmparam.BLOCK_M-1)/gemmparam.BLOCK_M);
            gemmparam.ASSIGN_Ntasks=((n+gemmparam.BLOCK_N-1)/gemmparam.BLOCK_N);
            gemmparam.ASSIGN_Ktasks=1;
            gemmparam.ASSIGN_batchtasks=batchCount;
            gemmparam.ASSIGN_ALLTASKS=gemmparam.ASSIGN_Mtasks*gemmparam.ASSIGN_Ntasks*gemmparam.ASSIGN_Ktasks*gemmparam.ASSIGN_batchtasks;

            gemmparam.ASSIGN_M=/*gemmparam.ASSIGN_Mtasks**/gemmparam.BLOCK_M;
            gemmparam.ASSIGN_N=/*gemmparam.ASSIGN_Ntasks**/gemmparam.BLOCK_N;
            gemmparam.ASSIGN_K=0;

            gemmparam.sharedmemory_size=sizeof(DEF_DOUBLE_C)*gemmparam.BLOCK_M*gemmparam.BLOCK_K;
            gemmparam.localmemory_size=sizeof(DEF_DOUBLE_AB) //要素当たりのサイズ
            *(((gemmparam.BLOCK_K+PACK_SIZE-1)/PACK_SIZE)*PACK_SIZE) //パックサイズの整数倍の行数のメモリを確保
            *MNB;
        }

        if(0)
        {
            std::cout << "gemmparam.BLOCK_M: " <<gemmparam.BLOCK_M<< std::endl;
            std::cout << "gemmparam.BLOCK_N: " <<gemmparam.BLOCK_N<< std::endl;
            std::cout << "gemmparam.BLOCK_K: " <<gemmparam.BLOCK_K<< std::endl;

            std::cout << "gemmparam.ASSIGN_Mtasks: " <<gemmparam.ASSIGN_Mtasks<< std::endl;
            std::cout << "gemmparam.ASSIGN_Ntasks: " <<gemmparam.ASSIGN_Ntasks<< std::endl;
            std::cout << "gemmparam.ASSIGN_Ktasks: " <<gemmparam.ASSIGN_Ktasks<< std::endl;
            std::cout << "gemmparam.ASSIGN_batchtasks: " <<gemmparam.ASSIGN_batchtasks<< std::endl;
            std::cout << "gemmparam.ASSIGN_ALLTASKS: " <<gemmparam.ASSIGN_ALLTASKS<< std::endl;

            std::cout << "gemmparam.ASSIGN_M: " <<gemmparam.ASSIGN_M<< std::endl;
            std::cout << "gemmparam.ASSIGN_N: " <<gemmparam.ASSIGN_N<< std::endl;
            std::cout << "gemmparam.ASSIGN_K: " <<gemmparam.ASSIGN_K<< std::endl;

            std::cout << "gemmparam.sharedmemory_size: " <<gemmparam.sharedmemory_size<< std::endl;
            std::cout << "gemmparam.localmemory_size: " <<gemmparam.localmemory_size<< std::endl;
        }
    }
    

    auto funcWrapper = single_BMNK_kernel::ssblasGemmBatchedEx_Batchgemm_single_BMNK_kernel<INDEXINT, T_SCALE, DEF_DOUBLE_AB, DEF_DOUBLE_C>; 
    const INDEXINT TASKNUM=gemmparam.ASSIGN_ALLTASKS;
    retval = ssblasEx::cpu::launcher::__SSBLASLAUNCH__(
        TASKNUM, 
        gemmparam.THREAD_PER_GROUP, 
        gemmparam.sharedmemory_size, 
        gemmparam.localmemory_size, 
        funcWrapper, 
            transa,
            transb,
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
            batchCount,
            gemmparam
    );

    return retval;
}


}
}
}
}
}



#endif