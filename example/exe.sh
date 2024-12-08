

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

# ./a.out int float float float N N 2 1.0  2568 782 3649 0
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

# https://www.ieice.org/~dpf/wp-content/uploads/2024/09/%E3%82%B9%E3%83%BC%E3%83%8F%E3%82%9A%E3%83%BC%E3%82%B3%E3%83%B3%E3%83%92%E3%82%9A%E3%83%A5%E3%83%BC%E3%82%BF%E3%80%8C%E5%AF%8C%E5%B2%B3%E3%80%8D%E3%81%A6%E3%82%99%E5%AD%A6%E7%BF%92%E3%81%97%E3%81%9F%E5%A4%A7%E8%A6%8F%E6%A8%A1%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%86%E3%82%99%E3%83%ABFugaku-LLM_v3.pdf
./a.out int float float float N N 6 1.0  144 2048 2048 0 2592 2048 144
./a.out int float float float N T 6 1.0  144 2048 2048 0 2048 2048 2048
./a.out int float float float N T 6 1.0  2048 144 2048 0 2048 2592 2048
./a.out int float float float T N 6 1.0  2048 2048 144 0 2592 2592 2048
./a.out int float float float T N 6 1.0  2048 2048 144 0 2592 864 2048