#include <immintrin.h>
#include "string.h"
#include <stdio.h>

const char* dgemm_desc = "Blocked dgemm with padding and AVX.";
// #pragma GCC optimize(3,"Ofast","inline","unroll-loops","prefetch-loop-arrays")

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#define BLOCK_SIZE2 64
#endif

#define BLOCK_L1 256
#define BLOCK_L2 512

#define Mat(A,i,j) (A)[(j)*lda + (i)]
#define min(a,b) (((a)<(b))?(a):(b))

// declare the functions
static void pad(double* restrict padA, double* restrict A, const int lda, const int newlda);
static void unpad(double* restrict padA, double* restrict A, const int lda, const int newlda);
void transposeA(double* restrict AT, double* restrict A, const int lda, const int ldb);
void copyA(double* restrict AT, double* restrict A, const int lda);
static void do_block_unrollavx128(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C);
static void do_block_unrollavx256(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C);
void square_dgemm (const int lda, double* restrict A, double* restrict B, double* restrict C);
void square_dgemm_365 (int lda, double* A, double* B, double*restrict C);
///copy array into the padded matrix
static void pad(double* restrict padA, double* restrict A, const int lda, const int newlda)
{
  for (int j = 0; j < lda; j++) {
    for (int i = 0; i < lda; i++) {
      padA[i + j*newlda] = A[i + j*lda];
    }   
  }   
}

//copy array into the unpadded matrix
static void unpad(double* restrict padA, double* restrict A, const int lda, const int newlda)
{
  for (int j = 0; j < lda; j++) {
    for (int i = 0; i < lda; i++) {
      A[i + j*lda] = padA[i + j*newlda];
    }   
  }   
}

//copy array into the unpadded matrix
// A is lda * ldb
void transposeA(double* restrict AT, double* restrict A, const int lda, const int ldb)
{
  for (int j = 0; j < ldb; j++) {
    for (int i = 0; i < lda; i++) {
      AT[j + i*ldb] = A[i + j*lda];
    }   
  }   
}

//copy array into the unpadded matrix
void copyA(double* restrict AT, double* restrict A, const int lda)
{
  for (int j = 0; j < lda; j++) {
    for (int i = 0; i < lda; i++) {
      AT[i + j*lda] = A[i + j*lda];
    }   
  }   
}

/* UNROLL 2: This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_unrollavx128(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C)
{
  __m128d Aik,Aik_1,Bkj,Bkj_1,Bkj_2,Bkj_3,Cij,Cij_1,d0,d1;

  for (int k = 0; k < K; k += 2 ) {
    for (int j = 0; j < N; j += 2) {
      Bkj = _mm_load1_pd(B+k+j*lda);
      Bkj_1 = _mm_load1_pd(B+k+1+j*lda);
      Bkj_2 = _mm_load1_pd(B+k+(j+1)*lda);
      Bkj_3 = _mm_load1_pd(B+k+1+(j+1)*lda);
      for (int i = 0; i < M; i += 2) {
        Aik = _mm_load_pd(A+i+k*lda);
        Aik_1 = _mm_load_pd(A+i+(k+1)*lda);

        Cij = _mm_load_pd(C+i+j*lda);
        Cij_1 = _mm_load_pd(C+i+(j+1)*lda);

        d0 = _mm_add_pd(Cij, _mm_mul_pd(Aik,Bkj));
        d1 = _mm_add_pd(Cij_1, _mm_mul_pd(Aik,Bkj_2));
        Cij = _mm_add_pd(d0, _mm_mul_pd(Aik_1,Bkj_1));
        Cij_1 = _mm_add_pd(d1, _mm_mul_pd(Aik_1,Bkj_3));
        _mm_store_pd(C+i+j*lda,Cij);
        _mm_store_pd(C+i+(j+1)*lda,Cij_1);
      }
    }
  }
}

/* UNROLL 2: This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_unrollavx256(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C)
{
  __m256d Aik, Aik_1, Bkj, Bkj_1, Bkj_2, Bkj_3, Cij, Cij_1, Cij_2, Cij_3, Cij_4, Cij_5, Cij_6, Cij_7;

  for (int i = 0; i < M; i += 8) {
    for (int j = 0; j < N; j += 4) {
      Cij = _mm256_load_pd(C+i+j*lda);
      Cij_1 = _mm256_load_pd(C+i+(j+1)*lda);
      Cij_2 = _mm256_load_pd(C+i+(j+2)*lda);
      Cij_3 = _mm256_load_pd(C+i+(j+3)*lda);

      Cij_4 = _mm256_load_pd(C+i+4+j*lda);
      Cij_5 = _mm256_load_pd(C+i+4+(j+1)*lda);
      Cij_6 = _mm256_load_pd(C+i+4+(j+2)*lda);
      Cij_7 = _mm256_load_pd(C+i+4+(j+3)*lda);

      for (int k = 0; k < K; ++k) {
        Aik = _mm256_load_pd(A+i+k*lda);
        Aik_1 = _mm256_load_pd(A+i+4+(k)*lda);

        Bkj = _mm256_broadcast_sd(B+k+j*lda);
        Bkj_1 = _mm256_broadcast_sd(B+k+(j+1)*lda);
        Bkj_2 = _mm256_broadcast_sd(B+k+(j+2)*lda);
        Bkj_3 = _mm256_broadcast_sd(B+k+(j+3)*lda);

        Cij = _mm256_add_pd(Cij, _mm256_mul_pd(Aik,Bkj));
        Cij_1 = _mm256_add_pd(Cij_1, _mm256_mul_pd(Aik,Bkj_1));
        Cij_2 = _mm256_add_pd(Cij_2, _mm256_mul_pd(Aik,Bkj_2));
        Cij_3 = _mm256_add_pd(Cij_3, _mm256_mul_pd(Aik,Bkj_3));

        Cij_4 = _mm256_add_pd(Cij_4, _mm256_mul_pd(Aik_1,Bkj));
        Cij_5 = _mm256_add_pd(Cij_5, _mm256_mul_pd(Aik_1,Bkj_1));
        Cij_6 = _mm256_add_pd(Cij_6, _mm256_mul_pd(Aik_1,Bkj_2));
        Cij_7 = _mm256_add_pd(Cij_7, _mm256_mul_pd(Aik_1,Bkj_3));
      }
      _mm256_store_pd(C+i+j*lda, Cij);
      _mm256_store_pd(C+i+(j+1)*lda, Cij_1);
      _mm256_store_pd(C+i+(j+2)*lda, Cij_2);
      _mm256_store_pd(C+i+(j+3)*lda, Cij_3);

      _mm256_store_pd(C+i+4+(j)*lda, Cij_4);
      _mm256_store_pd(C+i+4+(j+1)*lda, Cij_5);
      _mm256_store_pd(C+i+4+(j+2)*lda, Cij_6);
      _mm256_store_pd(C+i+4+(j+3)*lda, Cij_7);

    }
  }
}
void square_dgemm_256avx (const int lda, double* restrict A, double* restrict B, double* restrict C)
{
  // MODE = 0 -> use 128avx and unroll
  // MODE = 1 -> use 256avx and unroll
  int MODE = 1;

  int newlda = lda;
  int div = 8;
  if (lda % div){
    int t = lda % div;
    newlda = lda + (div-t) + div;
  }

  double* padA = (double*) _mm_malloc(newlda * newlda * sizeof(double), 32);
  pad(padA, A, lda, newlda);

  double* padB = (double*) _mm_malloc(newlda * newlda * sizeof(double), 32);
  pad(padB, B, lda, newlda);

  double* padC = (double*) _mm_malloc(newlda * newlda * sizeof(double), 32);
  pad(padC, C, lda, newlda);
  for (int k = 0; k < newlda; k += BLOCK_SIZE)
    /* For each block-row of A */ 
  {
    /* For each block-column of B */
    for (int j = 0; j < newlda; j += BLOCK_SIZE)
    {
      for (int i = 0; i < newlda; i += BLOCK_SIZE2)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min (BLOCK_SIZE2, newlda-i);
        int N = min (BLOCK_SIZE, newlda-j);
        int K = min (BLOCK_SIZE, newlda-k);
        /* Perform individual block dgemm */
        if (MODE == 0) do_block_unrollavx128(newlda, M, N, K, padA + i + k*newlda, padB + k + j*newlda, padC + i + j*newlda);
        else do_block_unrollavx256(newlda, M, N, K, padA + i + k*newlda, padB + k + j*newlda, padC + i + j*newlda);
      }
    }
  }
  unpad(padC, C, lda, newlda);
  _mm_free(padA);
  _mm_free(padB);
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (const int lda, double* restrict A, double* restrict B, double* restrict C)
{
  if(lda ==31||lda == 32||lda ==  97||lda ==  127||lda == 128||lda == 129||lda == 512){
    square_dgemm_365(lda, A, B, C);
  }
  else{
    square_dgemm_256avx(lda, A, B, C);
  }
}

static inline void matmul_4xkxkx4(int lda, int K, double* a, double* b, double* restrict c,int rM,int rN){
  __m256d a_coli, bi0, bi1, bi2, bi3;
  __m256d c_col0,c_col1,c_col2,c_col3;
  double *c_ptr1,*c_ptr2,*c_ptr3;

  c_ptr1=c+lda;
  c_ptr2=c_ptr1+lda;
  c_ptr3=c_ptr2+lda;
  c_col0 = _mm256_loadu_pd(c);
  c_col1 = _mm256_loadu_pd(c_ptr1);
  c_col2 = _mm256_loadu_pd(c_ptr2);
  c_col3 = _mm256_loadu_pd(c_ptr3);

  // for every column of a (or every row of b)
  for (int i = 0; i < K; ++i) 
  {
    a_coli = _mm256_load_pd(a);
    a += 4;

    bi0 = _mm256_broadcast_sd(b++);
    bi1 = _mm256_broadcast_sd(b++);
    bi2 = _mm256_broadcast_sd(b++);
    bi3 = _mm256_broadcast_sd(b++);
    c_col0 = _mm256_fmadd_pd(a_coli, bi0,c_col0);
    c_col1 = _mm256_fmadd_pd(a_coli, bi1,c_col1);
    c_col2 = _mm256_fmadd_pd(a_coli, bi2,c_col2);
    c_col3 = _mm256_fmadd_pd(a_coli, bi3,c_col3);
  }
  if (rM==4&&rN==4){
      _mm256_storeu_pd(c,c_col0);
      _mm256_storeu_pd(c_ptr1,c_col1);
      _mm256_storeu_pd(c_ptr2,c_col2);
      _mm256_storeu_pd(c_ptr3,c_col3);
    }
  else{ 
    double ctem[4*4];
    _mm256_storeu_pd(ctem, c_col0);
    _mm256_storeu_pd(ctem+4, c_col1);
    _mm256_storeu_pd(ctem+8, c_col2);
    _mm256_storeu_pd(ctem+12, c_col3);
    for (int i=0;i<rM;i++){
      for (int j=0;j<rN;j++){
      	c[i+j*lda]=ctem[i+j*4];
      }
    }
  }
}
static inline void copy_a (int lda, const int K, double* a_src, double* a_dst) {
  for (int i = 0; i < K; ++i) 
  {
    for ( int j = 0; j < 4; ++j){
      *a_dst++ = a_src[j];
    }
    a_src += lda;
  }
}
static inline void copyadd0_a (int lda, const int K, double* a_src, double* a_dst,int rc) {
  for (int i = 0; i < K; ++i) 
  {
    for ( int j = 0; j < 4; ++j){
      if (j>=rc){
        *a_dst=0;
	      a_dst++;
      }
      else{
        *a_dst = a_src[j];
        a_dst++;
      }
    }
    a_src += lda;
  }
}
// copy and transpose B
static inline void copy_b (int lda, const int K, double* b_src, double* b_dst) {
  double* b_ptr[4];
  b_ptr[0] = b_src;
  for(int i = 1; i < 4; ++i){
    b_ptr[i] = b_ptr[i-1] + lda;
  }

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < 4; ++j){
      *b_dst++ = *b_ptr[j]++;
    }
  }
}
static inline void copyadd0_b (int lda, const int K, double* b_src, double* b_dst,int rc) {
  double* b_ptr[4];
  b_ptr[0] = b_src;
  for(int i = 1; i < rc; ++i){
    b_ptr[i] = b_ptr[i-1] + lda;
  }

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < 4; ++j){
      if (j>=rc){
        *b_dst=0;
      	b_dst++;
      }
      else{
      *b_dst++ = *b_ptr[j]++;
      }
    }
  }

}
/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block (int lda, int M, int N, int K,int newM,int newN, double* A, double* B, double* restrict C, double* A_block, double* B_block)
{
  double *a_ptr, *b_ptr;
  /* For each column of B */
  for (int j = 0 ; j < newN; j += 4) 
  {
    b_ptr = &B_block[K * j];
    if (j>N-4) {copyadd0_b(lda,K,B+j*lda,b_ptr,N-j);}
    else{// copy and transpose B_block
    copy_b(lda, K, B + j*lda, b_ptr);}

    for (int i = 0; i < newM; i += 4) {
      a_ptr = &A_block[i * K];
      if (j == 0){
        if (i>M-4){copyadd0_a(lda,K,A+i,a_ptr,M-i);}
        else{copy_a(lda, K, A + i, a_ptr);}
      }
      int rM=min(4,M-i);
      int rN=min(4,N-j);
      matmul_4xkxkx4(lda, K, a_ptr, b_ptr, C+i+j*lda,rM,rN);
    }
  }
}
 void square_dgemm_365 (int lda, double* A, double* B, double*restrict C)
{
  double *A_block, *B_block;  
  int BLOCK_L2M = 512;
  int BLOCK_L2N = 512;
  int BLOCK_L2K = 256;
  int BLOCK_L1M = 256;
  int BLOCK_L1N = 256;
  int BLOCK_L1K = 128;
  posix_memalign((void **)&A_block, 32, BLOCK_L1 * BLOCK_L1 * sizeof(double));
  posix_memalign((void **)&B_block, 32, BLOCK_L1 * BLOCK_L1 * sizeof(double));

  for (int k1 = 0; k1 < lda; k1 += BLOCK_L2K){
    for (int j1 = 0; j1 < lda; j1 += BLOCK_L2N){
      for (int i1 = 0; i1 < lda; i1 += BLOCK_L2M){
        int end_k = k1 + min(BLOCK_L2K, lda-k1);
        int end_j = j1 + min(BLOCK_L2N, lda-j1);
        int end_i = i1 + min(BLOCK_L2M, lda-i1);
        for (int k = k1; k < end_k; k += BLOCK_L1K){
          for (int j = j1; j < end_j; j += BLOCK_L1N){
            for (int i = i1; i < end_i; i += BLOCK_L1M){
              int K = min(BLOCK_L1K, end_k-k);
              int N = min(BLOCK_L1N, end_j-j);
              int M = min(BLOCK_L1M, end_i-i);
              int resrow = M % 4;
              int rescol = N % 4;
              int newM=M;
              int newN=N;
              if (resrow>0){
                newM=M+4-resrow;
              }
              if (rescol>0){
                newN=N+4-rescol;
              }
              do_block(lda, M, N, K, newM,newN,A + i + k*lda, B + k + j*lda, C + i + j*lda, A_block, B_block);
            }
          }
        }
      }
    }
  }
  free(A_block);
  free(B_block);
}