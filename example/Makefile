FCC_CHECK := $(shell command -v FCC > /dev/null 2>&1 && echo "FOUND" || echo "NOTFOUND")
FCCpx_CHECK := $(shell command -v FCCpx > /dev/null 2>&1 && echo "FOUND" || echo "NOTFOUND")
aarch64-linux-gnu-g++-12_CHECK := $(shell command -v aarch64-linux-gnu-g++-12 > /dev/null 2>&1 && echo "FOUND" || echo "NOTFOUND")
g++_CHECK := $(shell command -v g++ > /dev/null 2>&1 && echo "FOUND" || echo "NOTFOUND")

ifeq ($(g++_CHECK),FOUND)
    CXX = g++ -fsigned-char
    CXX_OPT = -O3
endif
ifeq ($(aarch64-linux-gnu-g++-12_CHECK),FOUND)
    CXX = aarch64-linux-gnu-g++-12  -march=armv8.2-a+sve -fsigned-char
    CXX_OPT = -O3
endif
ifeq ($(FCC_CHECK),FOUND)
    CXX = FCC -Nclang -fsigned-char -std=gnu++17
    CXX_OPT = -Ofast
endif
ifeq ($(FCCpx_CHECK),FOUND)
    CXX = FCCpx -Nclang -fsigned-char -std=gnu++17
    CXX_OPT = -Ofast
endif



BATCHLIB = -L../src -lssblasGemmBatchedEx
OMP = -fopenmp

HPPSRCS = $(wildcard include/*.hpp) \


include Makefile.blas

a.out: main.cpp $(HPPSRCS)
	$(CXX) main.cpp $(CXX_OPT) $(BATCHLIB) $(OMP) $(LIB_BLAS) $(INC_BLAS)