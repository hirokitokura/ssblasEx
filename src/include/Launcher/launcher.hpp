#ifndef SSBLAS_LAUNCHER
#define SSBLAS_LAUNCHER
#include<stdio.h>
#include<string.h>
#include<iostream>
#include <functional>
#include <omp.h>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <cassert>
#include "../../../include/ssblasBatch.h"

namespace ssblasEx{
namespace cpu{
namespace launcher{

class ssblasBarrier_t{
	
	
	
	public:
	int THREAD_NUM;
	volatile int wait_counter;
	std::condition_variable_any  cv;
	std::mutex mtx;

	omp_lock_t omp_lock; 

	int* counter;
	volatile int shared_counter;

	ssblasBarrier_t(int THREAD_NUM): THREAD_NUM(THREAD_NUM), wait_counter(0), shared_counter(0){
		//wait_counter=0;
		omp_init_lock(&omp_lock);

		counter=(int*)malloc(sizeof(int)*THREAD_NUM);
		for(int i=0;i<THREAD_NUM;i++){
			counter[i]=0;
		}
	}
	
	void barrier(int threadIdx){
		counter[threadIdx]+=1;
		counter[threadIdx]=counter[threadIdx]%2;
		#pragma omp flush
		if(threadIdx==0){
			constexpr int THREADIDX_ZERO=0;
			for(int i=1;i<THREAD_NUM;i++){
				volatile int local_counter=counter[i];
				#pragma omp flush
				while(counter[THREADIDX_ZERO]!=local_counter){
					//printf("loop\n");
					local_counter=counter[i];
					#pragma omp flush
				}
			}
			shared_counter=counter[THREADIDX_ZERO];
			#pragma omp flush
		}
		volatile int local_counter=shared_counter;
		while(shared_counter!=counter[threadIdx]){
			#pragma omp flush
			local_counter=shared_counter;
			#pragma omp flush
		}
		//printf("%d end barrier\n", omp_get_thread_num());
	}
	/*
	void barrier(){
		std::unique_lock<std::mutex> uniq_lk(mtx);

		//if (wait_counter == THREAD_NUM) wait_counter = 0;
		omp_set_lock(&omp_lock);
		wait_counter++;
		omp_unset_lock(&omp_lock);
		#pragma omp flush
		//printf("count:%d wait_counter:%d %d\n", count, (int)wait_counter, omp_get_thread_num());
		if (wait_counter == THREAD_NUM){
			printf("notify_all count:%d wait_counter:%d %d\n", wait_counter, (int)wait_counter, omp_get_thread_num());
			omp_set_lock(&omp_lock);
			wait_counter=0;
			omp_unset_lock(&omp_lock);
			#pragma omp flush
			cv.notify_all();
			#pragma omp flush
		}
		else{
			printf("count:%d wait_counter:%d %d\n", wait_counter, (int)wait_counter, omp_get_thread_num());
			cv.wait(uniq_lk, [this] (){ 
				return wait_counter == 0; });
		printf("count:%d wait_counter:%d %d\n", wait_counter, (int)wait_counter, omp_get_thread_num());
		}
		
		printf("%d end barrier\n", omp_get_thread_num());
		#pragma omp flush
	}
	*/
	~ssblasBarrier_t(){
		omp_destroy_lock(&omp_lock);
		free(counter);
	}
};

class ssblasIndex_t{
	using ssblasIndex_t_type=long int;
	public:
	const ssblasIndex_t_type threadIdx=0;//グループ内のスレッド番号(0,1,2,..)
	const ssblasIndex_t_type groupSize=0;//グループあたりのスレッド数
	const ssblasIndex_t_type groupIdx=0;//グループのインデックス番号(0,1,2,...)
	const ssblasIndex_t_type launchgroupSize=0;//このカーネル中に存在するgroupの数
	const ssblasIndex_t_type residentgroupSize=0;//このカーネル中に実際に存在する有効なgroupの数	
	
	ssblasIndex_t(
		ssblasIndex_t_type threadIdx, 
		ssblasIndex_t_type groupSize,
		ssblasIndex_t_type groupIdx,
		ssblasIndex_t_type launchgroupSize,
		ssblasIndex_t_type residentgroupSize)
		: threadIdx(threadIdx),
		groupSize(groupSize),
		groupIdx(groupIdx),
		launchgroupSize(launchgroupSize),
		residentgroupSize(residentgroupSize){}
		
	void show(){
		#pragma omp critical
		{
			printf("*********\n");
			printf("threadIdx: %ld\n", threadIdx);
			printf("groupSize: %ld\n", groupSize);
			printf("groupIdx: %ld\n", groupIdx);
			printf("launchgroupSize: %ld\n", launchgroupSize);
			printf("residentgroupSize: %ld\n", residentgroupSize);
			printf("*********\n");
		}
	}

};

class ssblasKernelHandle_t{
	public:
	ssblasIndex_t* ssblasIndex;
	ssblasBarrier_t* ssblasBarrier;
	void* shared_ptr;
	void* local_ptr;
	
	ssblasKernelHandle_t(ssblasIndex_t* ssblasIndex, ssblasBarrier_t* ssblasBarrier, void* shared_ptr, void* local_ptr)
	:ssblasIndex(ssblasIndex), ssblasBarrier(ssblasBarrier), shared_ptr(shared_ptr), local_ptr(local_ptr){
	}
	
	void barrier(){
		ssblasBarrier->barrier(ssblasIndex->threadIdx);
	}

	void show(){
		ssblasIndex->show();
		printf("shared_ptr: %p\n", shared_ptr);
	}
};



//このカーネルはライブラリ内部で用いるカーネルランチャー
template<class FUNCTYPE, typename... Args> 
ssblasStatus_t __SSBLASLAUNCH__ (
	long int BLOCKNUM, 
	long int THREADNUM, 
	size_t sharedmemory_size, 
	size_t localmemory_size, 
	FUNCTYPE FUNC, Args... args)
{
    ssblasStatus_t retval=SSBLAS_STATUS_SUCCESS;
	if(!(BLOCKNUM>0)){
		return SSBLAS_STATUS_INTERNAL_ERROR;
	}
	if(!(THREADNUM>0)){
		return SSBLAS_STATUS_INTERNAL_ERROR;
	}
	int TOTAL_THREAD_NUM;
	#pragma omp parallel
	{
		TOTAL_THREAD_NUM=omp_get_num_threads();
	}
	if(TOTAL_THREAD_NUM%THREADNUM!=0){
		return SSBLAS_STATUS_INTERNAL_ERROR;
	}
    if(ssblasEx::cpu::utils::MAX_THREADTEAM_NUM<TOTAL_THREAD_NUM){
        return SSBLAS_STATUS_INTERNAL_ERROR;
    }
    //std::cout << "TOTAL_THREAD_NUM: " << TOTAL_THREAD_NUM << std::endl;
    int group_num=TOTAL_THREAD_NUM/THREADNUM;
    if(!(group_num>0)){
		return SSBLAS_STATUS_INTERNAL_ERROR;
	}
	ssblasBarrier_t** BAR = new ssblasBarrier_t*[ssblasEx::cpu::utils::MAX_THREADTEAM_NUM];
	for (int i = 0; i < group_num; ++i) { 
        BAR[i] = new ssblasBarrier_t(THREADNUM);
    }
	
	void* sharedmemory_ptr[ssblasEx::cpu::utils::MAX_THREADTEAM_NUM];
	for (int i = 0; i < group_num; ++i) { 
        sharedmemory_ptr[i]=ssblasEx::cpu::utils::ssblas_malloc(sharedmemory_size);
	}
	#pragma omp parallel 
    {
		void* localmemory_ptr=NULL;
		localmemory_ptr=ssblasEx::cpu::utils::ssblas_malloc(localmemory_size);

		int local_id=omp_get_thread_num()%THREADNUM;
		int group_id=omp_get_thread_num()/THREADNUM;
		//std::cout << "TOTAL_THREAD_NUM: " << TOTAL_THREAD_NUM << std::endl;
		
		//printf("BLOCKNUM: %d\n",BLOCKNUM);
		//printf("local_id: %d, group_id: %d\n",local_id, group_id);
		
		
		
		for(int task_id=group_id;task_id<BLOCKNUM;task_id+=group_num){
			ssblasIndex_t ssblasIndex(local_id, THREADNUM, task_id, BLOCKNUM, group_num);
			ssblasKernelHandle_t KernelHandle(&ssblasIndex, BAR[group_id], sharedmemory_ptr[group_id], localmemory_ptr);
			/*printf("local_id:%d group_id: %d omp_get_thread_num:%d\n", 
				KernelHandle.ssblasIndex->threadIdx, 
				KernelHandle.ssblasIndex->groupIdx, 
				omp_get_thread_num());*/
			ssblasStatus_t result=FUNC(KernelHandle,args...);
            if(SSBLAS_STATUS_SUCCESS!=result){
                retval=result;
            }
			//std::cout << result << std::endl;
			//printf("result: %f\n",result);
		}

		ssblasEx::cpu::utils::ssblas_free(localmemory_ptr);
	}
	
	
	for (int i = 0; i < group_num; ++i) {
        delete BAR[i]; 
    }
	delete[] BAR;
	for (int i = 0; i < group_num; ++i) { 
		ssblasEx::cpu::utils::ssblas_free(sharedmemory_ptr[i]);
	};
	return retval;
}




/*
template<typename T0,typename T1>
int func1(ssblasKernelHandle_t kernelhandle,T0 a, T1 b)
{
	std::cout << a << std::endl;
	kernelhandle.ssblasBarrier->barrier();
	kernelhandle.ssblasIndex->show();
	std::cout << b << std::endl;
	kernelhandle.show();
	return (int)(a+b);
}

float func2(ssblasKernelHandle_t kernelhandle,float a, float b)
{
	std::cout << a << std::endl;
	kernelhandle.ssblasBarrier->barrier();
	kernelhandle.ssblasIndex->show();
	std::cout << b << std::endl;
	kernelhandle.show();
	return (float)(a+b);
}

int main(void)
{
	//std::function<int(ssblasBarrier_t*, int, double)> funcWrapper = func1<int, double>; // 関数マクロを使用してテンプレート関数を呼び出し、戻り値を取得 
	std::function<float(ssblasKernelHandle_t, float, float)> funcWrapper = func2; // 関数マクロを使用してテンプレート関数を呼び出し、戻り値を取得 
	int result = __SSBLASLAUNCH__(2, 4, sizeof(float)*1024, funcWrapper, 9.0f, 2.3f); 
	// 結果を出力 
	std::cout << "Result: " << result << std::endl;
	return 0;
}
*/
}
}
}

#endif