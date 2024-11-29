# ssblasEx
このライブラリ(Simple Simd blas Experimental)は、様々な型の行列積を行うことを目的にした実験的なライブラリです。
SVEアーキテクチャを前提にしています。


# データタイプ

## ssblasStatus_t

このタイプは、関数の実行結果状態を返すために使われる。
| Error Value  | Meaning | 
| ------------- | ------------- |
| SSBLAS_STATUS_SUCCESS | 計算は正しく完了した。 |
| SSBLAS_STATUS_ALLOC_FAILED | 内部でメモリ確保に失敗した。 |
| SSBLAS_STATUS_INVALID_VALUE | 行列積の引数が不適切であった。 |
| SSBLAS_STATUS_INTERNAL_ERROR | 内部エラー |
| SSBLAS_STATUS_NOTIMPLEMENTED_ERROR | 実装していないパラメータの組み合わせだった。 |


## ssblasOperation_t

このタイプは、行列を転置するかどうかを指定する。

|  Value  | Meaning | 
| ------------- | ------------- |
| SSBLAS_OP_N | 転置しない。 |
| SSBLAS_OP_T | 転置する。 |
| SSBLAS_OP_C | 共役転置をする。 |

## ssblasDataType_t

このタイプは、行列やスケール値の型を指定する。

|  Value  | Meaning | 
| ------------- | ------------- |
| SSBLAS_R_8I | signed char型 |
| SSBLAS_R_32I | signed int 型 |
| SSBLAS_R_32F | float型 |
| SSBLAS_R_64F | double型 |


## ssblasComputeType_t

このタイプは、行列積における計算精度を指定する。

|  Value  | Meaning | 
| ------------- | ------------- |
| SSBLAS_COMPUTE_DEFAULT_TYPE | デフォルトの値で計算をする。 |


## ssblasGemmAlgo_t

このタイプは、行列積におけるアルゴリズムを指定する。

|  Value  | Meaning | 
| ------------- | ------------- |
| SSBLAS_COMPUTE_DEFAULT | デフォルトのアルゴリズムで計算をする。 |


# 提供関数

## cublasGemmBatchedEx

```
ssblasStatus_t ssblasGemmBatchedEx(
    ssblasOperation_t    transa,
    ssblasOperation_t    transb,
    int                  m,
    int                  n,
    int                  k,
    const void*          alpha,
    const void *const    A[],
    ssblasDataType_t     Atype,
    const int            lda,
    const void *const    B[],
    ssblasDataType_t     Btype,
    const int            ldb,
    const void*          beta,
    void *const          C[],
    ssblasDataType_t     Ctype,
    const int            ldc,
    const int            batchCount,
    ssblasComputeType_t  computeType,
    ssblasGemmAlgo_t     algo
);
```

この関数は、以下のようなバッチ行列積を行う関数である。

$$ C[i] = \alpha op(A[i])op(B[i]) + \beta C[i] $$

$\alpha$ と $\beta$ はスカラーである。
$A$, $B$, $C$ は行列へのポインタの配列である。行列は列優先で保存されている。

$op(A[i])$ の次元は $m \times k$ ,
$op(B[i])$ の次元は $k \times n$ ,
$C[i]$ の次元は $m \times n$ である。

また、行列 $A$ に対する $op(A)$ は以下のようになる。

$$ 
\begin{cases}
A \text{ if } transa == SSBLAS \\_ OP \\_ N \\
A^T \text{ if } transa == SSBLAS \\_OP \\_T \\
A^H \text{ if } transa == SSBLAS\\_OP\\_C
\end{cases}
$$

行列 $B$ に対しても同様である。


| Param.  | In/out | Meaning |
| ------------- | ------------- | ------------- |
| transa  | input  | $Op(A)$ で転置するかの指定 |
| transb  | input  | $Op(B)$ で転置するかの指定 |
| m  | input  | $Op(A)$ と $C$ の行数 |
| n  | input  | $Op(A)$ と $C$ の列数 |
| k  | input  | $Op(A)$ の列数と $Op(B)$ の行数 |
| alpha  | input  | $A \times B$ のスケール値。対応するcomputeTypeを指定する必要がある。詳細は別表を参照|
| A  | input  | 行列Aに対応するポインタ配列 |
| Atype  | input  | 行列Aのデータ型を示す列挙型 |
| lda  | input  | $A[i]$ が保存されている行列のリーディングディメンション |
| B  | input  | 行列Bに対応するポインタ配列 |
| Btype  | input  | 行列Bのデータ型を示す列挙型 |
| ldb  | input  | $B[i]$ が保存されている行列のリーディングディメンション |
| beta  | input  | 行列 $C$ に対するスケール値。 $beta==0$ の場合は $C[i]$ は有効な入力である必要はない。 |
| C  | in/out  | 行列Cに対応するポインタ配列 |
| Ctype  | input  | 行列Cのデータ型を示す列挙型 |
| ldc  | input  | $C[i]$ が保存されている行列のリーディングディメンション |
| batchCount  | input  | 行列積数 |
| computeType  | input  | 計算タイプを指定する列挙型 |
| algo  | input  | アルゴリズムを指定する列挙型 |


ssblasGemmBatchedExは以下の計算型、スケール型、Atype/Btype/Ctypeをサポートする。


| Compute Type  | Scale Type (alpha and beta) | Atype/Btype | Ctype |
| ------------- | ------------- | ------------- | ------------- | 
| SSBLAS_COMPUTE_DEFAULT_TYPE | SSBLAS_R_32F | SSBLAS_R_32F | SSBLAS_R_32F |
| SSBLAS_COMPUTE_DEFAULT_TYPE | SSBLAS_R_64F | SSBLAS_R_64F | SSBLAS_R_64F |
| SSBLAS_COMPUTE_DEFAULT_TYPE | SSBLAS_R_8I | SSBLAS_R_8I | SSBLAS_R_32I |


以下の表は、ssblasGemmBatchedExが返す値とそれらの意味を示す。

| Error Value  | Meaning | 
| ------------- | ------------- |
| SSBLAS_STATUS_SUCCESS | 計算は正しく完了した。 |
| SSBLAS_STATUS_ALLOC_FAILED | 内部でメモリ確保に失敗した。 |
| SSBLAS_STATUS_INVALID_VALUE | 行列積の引数が不適切であった。 |
| SSBLAS_STATUS_INTERNAL_ERROR | 内部エラー |
| SSBLAS_STATUS_NOTIMPLEMENTED_ERROR | 実装していないパラメータの組み合わせだった。 |

なお、mが負といったエラーチェックは現在行っていない。


# コンパイルとexampleの実行

## ライブラリのコンパイル方法
```
cd src
make
```
正しくコンパイルされれば`libssblasGemmBatchedEx.so`が生成されます。
用いる際は、LD_LIBRARY_PATHへパスの追加をしてください。

また、必要に応じて
```
Makefile.comp
```
内部のオプションなどを変更してください。

## exampleのコンパイルと実行
事前にOpenBLASなどのライブラリを用意してください。

```
cd example
make
./exe.sh
```

### exampleの引数
生成された`a.out`の引数は以下のように設定してください。
```
./a.out int <スケール値の型> <行列A,Bの型> <行列Cの型> <行列Aの転置の有無> <行列Bの転置の有無> <バッチ数> <alphaの値> <M> <N> <K> <betaの値>
```


# 環境変数

## SSBLAS_GEMMBATCHEDEX_DEBUG

```
SSBLAS_GEMMBATCHEDEX_DEBUG
```

この環境変数が設定されているとき、ssblasGemmBatchedExの引数の情報を表示する。
出力例

```
ssblasGemmBatchedExShowdebug: START
transa: SSBLAS_OP_T
transb: SSBLAS_OP_T
M: 1024
N: 1024
K: 1024
alpha: 
A: 
Atype: SSBLAS_R_8I
lda: 1024
B: 
Btype: SSBLAS_R_8I
ldb: 1024
beta: 
C: 
Ctype: SSBLAS_R_32I
ldc: 1024
batchCount: 3
computeType: SSBLAS_COMPUTE_DEFAULT_TYPE
algo: SSBLAS_COMPUTE_DEFAULT
ssblasGemmBatchedExShowdebug: END
```


# 補足

このライブラリは以下のリポジトリのファイルを改変し、使用しています。
https://github.com/fujitsu/OpenBLAS/tree/fj_develop

# License
See [LICENSE](https://github.com/hirokitokura/ssblas/blob/main/LICENSE) file.
