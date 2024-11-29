

cd ../
cd src
make 
cd ../
cd example
make
BATCH_LIB=../src
export LD_LIBRARY_PATH=${BATCH_LIB}:${LD_LIBRARY_PATH}

BLAS_LIB=~/local/openblas_arm/
#BLAS_LIB=~/local/openblas_intel/
export LD_LIBRARY_PATH=${BLAS_LIB}/lib:${LD_LIBRARY_PATH}


ulimit -s unlimited
ldd a.out
export OMP_NUM_THREADS=12
export SSBLAS_GEMMBATCHEDEX_DEBUG=1
#./a.out int int char int  N N 1 1.0  8 1 8 0
#./a.out int float float float N N 1 1.0  8 12 8 0
#./a.out int float float float N N 1 1.0  8 12 513 0
#./a.out int float float float N N 1 1.0  129 12 513 0
#./a.out int float float float N N 2 1.0  512 512 256 0
#./a.out int float float float N N 2 1.0  512 512 512 0
#./a.out int float float float N N 1 1.0  512 512 1024 0
#./a.out int double double double N N 4 1.0  512 512 256 0
#./a.out int int char int N N 5 1.0  512 512 256 0
#./a.out int float float float T N 2 1.0  512 512 256 0
#./a.out int float float float N T 2 1.0  512 512 256 0
#./a.out int float float float T T 2 1.0  512 512 256 0

#./a.out int float float float N N 1 1.0  1000 1000 1000 0
#./a.out int float float float N N 1 1.0  12300 12300 12300 0

./a.out int float float float N N 2 1.0  2568 782 3649 0
./a.out int float float float N N 2 1.0  2568 782 3649 0
./a.out int float float float T N 2 1.0  2568 782 3649 0
./a.out int float float float N T 2 1.0  2568 782 3649 0
./a.out int float float float T T 2 1.0  2568 782 3649 0
./a.out int double double double N N 2 1.0  2568 782 3649 0
./a.out int double double double T N 2 1.0  2568 782 3649 0
./a.out int double double double N T 2 1.0  2568 782 3649 0
./a.out int double double double T T 2 1.0  2568 782 3649 0
./a.out int int char int  N N 2 1.0  2568 782 3649 0
./a.out int int char int  T N 2 1.0  2568 782 3649 0
./a.out int int char int  N T 2 1.0  2568 782 3649 0
./a.out int int char int  T T 2 1.0  2568 782 3649 0

#./a.out int float float float N N 2 1.0  1024 1024 1024 0
#./a.out int float float float T N 1 1.0  1024 1024 1024 0
#./a.out int float float float N T 1 1.0  1024 1024 1024 0
#./a.out int float float float T T 1 1.0  1024 1024 1024 0
#./a.out int float float float N T 1 1.0  1 2  1 0
#./a.out int int char int N T 1 1.0  1024 1024 1024 0
#./a.out int int char int T N 1 1.0  1024 1024 1024 0
#./a.out int int char int T T 1 1.0  1024 1024 1024 0
#./a.out int double double double T T 1 1.0  1024 1024 1024 0
#./a.out int double double double N N 1 1.0  1024 1024 1024 0
#./a.out int float float float N N 1 1.0  1024 1024 1024 0
#./a.out int double double double T T 1 1.0  2 3 5 0
#./a.out int int char int N N 1 1.0  64 64 1025 0
#./a.out int int char int T N 1 1.0  64 64 1025 0


#./a.out int float float float N N 3 1.0  128 128    1025 0
#./a.out int float float float N T 1 1.0  128 128    1025 0
#./a.out int float float float T N 1 1.0  128 128    1025 0
#./a.out int float float float T T 1 1.0  128 128    1025 0
#./a.out int double double double N N 3 1.0  128 128    1025 0
#./a.out int double double double N T 1 1.0  128 128    1025 0
#./a.out int double double double T N 1 1.0  128 128    1025 0
#./a.out int double double double T T 1 1.0  128 128    1025 0
#./a.out int int char int N N 3 1.0  128 128    1025 0
#./a.out int int char int N T 1 1.0  128 128    1025 0
#./a.out int int char int T N 1 1.0  128 128    1025 0
#./a.out int int char int T T 1 1.0  128 128    1025 0



#./a.out int int char int N N 1 1.0  1 2      1 0
#./a.out int int char int N T 1 1.0  1 2      1 0
#./a.out int int char int T T 1 1.0  1 2      1 0

#./a.out int float float float N N 1 1.0  65 65 1025 0
#./a.out int float float float N N 1 1.0  8 8 4 0
#./a.out int double double double T N 1 1.0  65 65 1025 0
#./a.out int double double double N N 1 1.0  128 128 2048 0
#./a.out int int char int N N 1 1.0  1 1 8 0
#./a.out int int char int N T 1 1.0  1 1 8 0
#./a.out int int char int T N 1 1.0  1 1 8 0
#./a.out int float float float T N 1 1.0  256  256  257 0
#./a.out int double double double T N 1 1.0  256  256  257 0
#./a.out int int char int N N 1 1   599 688 1867 1