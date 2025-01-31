#include <immintrin.h>
#include "string.h"
#include <stdio.h>

const char* dgemm_desc = "Blocked dgemm with padding and AVX.";

#pragma GCC optimize(3,"Ofast","inline","unroll-loops","prefetch-loop-arrays")
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#define BLOCK_SIZE2 64
#endif

#define min(a,b) (((a)<(b))?(a):(b))

// declare the functions
static void pad(double* restrict padA, double* restrict A, const int lda, const int newlda);
static void unpad(double* restrict padA, double* restrict A, const int lda, const int newlda);
void transposeA(double* restrict AT, double* restrict A, const int lda, const int ldb);
void copyA(double* restrict AT, double* restrict A, const int lda);
static void do_block (const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C);
static void do_block_unrollavx128(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C);
static void do_block_unrollavx256(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C);
void square_dgemm (const int lda, double* restrict A, double* restrict B, double* restrict C);

///copy array into the padded matrix
static void pad(double* restrict padA, double* restrict A, const int lda, const int newlda)
{ double * Ap,*Apadp;
  Ap=A;
  Apadp=padA-newlda+lda;
  for (int j = 0; j < lda; j++) {
    Apadp+=-lda+newlda;
    for (int i = 0; i < lda; i++) {
      *Apadp=*Ap;
      Ap++;
      Apadp++;
    }   
  }   
}
static inline void padT(double*restrict padB, double* restrict B,const int lda,const int newlda)
{
  for (int i=0;i<lda;i++)
  {
    for (int j=0;j<lda;j++)
    {
      padB[i+j*newlda]=B[j+i*lda];
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

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
  {
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
      for (int k = 0; k < K; ++k){
        cij += A[i+k*lda] * B[k+j*lda];
        C[i+j*lda] = cij;
      }
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
static inline void do_block_unrollavx256(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C)
{
  __m256d Aik, Aik_1,Aik_2, Bkj, Bkj_1, Bkj_2, Bkj_3, Cij, Cij_1, Cij_2, Cij_3, Cij_4, Cij_5, Cij_6, Cij_7,Cij_8,Cij_9,Cij_10,Cij_11;
double *Bs,*Cs1,*Cs2,*Cs3,*Cs4,*Cs5,*Cs6,*Cs7,*Cs8,*As;
  for (int i = 0; i < M; i +=8 ) {
    for (int j = 0; j < N; j += 4) {
      Cs1=C+i+j*lda;
      Cij = _mm256_load_pd(Cs1);
      Cs2=Cs1+lda;
      Cij_1 = _mm256_load_pd(Cs2);
      Cs3=Cs2+lda;
      Cij_2 = _mm256_load_pd(Cs3);
      Cs4=Cs3+lda;
      Cij_3 = _mm256_load_pd(Cs4);
      Cs5=Cs1+4;
      Cij_4 = _mm256_load_pd(Cs5);
      Cs6=Cs5+lda;
      Cij_5 = _mm256_load_pd(Cs6);
      Cs7=Cs6+lda;
      Cij_6 = _mm256_load_pd(Cs7);
      Cs8=Cs7+lda;
      Cij_7 = _mm256_load_pd(Cs8);

 //     Cij_8 = _mm256_load_pd(C+i+8+j*lda);
  //    Cij_9 = _mm256_load_pd(C+i+8+(j+1)*lda);
   //   Cij_10 = _mm256_load_pd(C+i+8+(j+2)*lda);
    //  Cij_11 = _mm256_load_pd(C+i+8+(j+3)*lda);
      for (int k = 0; k < K; ++k) {
        As=A+i+k*lda;
        Aik = _mm256_load_pd(As);
	As+=4;
        Aik_1 = _mm256_load_pd(As);
      //  Aik_2 = _mm256_load_pd(A+i+8+(k)*lda);
        Bs=B+k+j*lda;
        Bkj = _mm256_broadcast_sd(Bs);
	Bs+=lda;
        Bkj_1 = _mm256_broadcast_sd(Bs);
	Bs+=lda;
        Bkj_2 = _mm256_broadcast_sd(Bs);
	Bs+=lda;
        Bkj_3 = _mm256_broadcast_sd(Bs);

        Cij = _mm256_add_pd(Cij, _mm256_mul_pd(Aik,Bkj));
        Cij_1 = _mm256_add_pd(Cij_1, _mm256_mul_pd(Aik,Bkj_1));
        Cij_2 = _mm256_add_pd(Cij_2, _mm256_mul_pd(Aik,Bkj_2));
        Cij_3 = _mm256_add_pd(Cij_3, _mm256_mul_pd(Aik,Bkj_3));

        Cij_4 = _mm256_add_pd(Cij_4, _mm256_mul_pd(Aik_1,Bkj));
        Cij_5 = _mm256_add_pd(Cij_5, _mm256_mul_pd(Aik_1,Bkj_1));
        Cij_6 = _mm256_add_pd(Cij_6, _mm256_mul_pd(Aik_1,Bkj_2));
        Cij_7 = _mm256_add_pd(Cij_7, _mm256_mul_pd(Aik_1,Bkj_3));

       // Cij_8 = _mm256_add_pd(Cij_8, _mm256_mul_pd(Aik_2,Bkj));
       // Cij_9 = _mm256_add_pd(Cij_9, _mm256_mul_pd(Aik_2,Bkj_1));
       // Cij_10 = _mm256_add_pd(Cij_10, _mm256_mul_pd(Aik_2,Bkj_2));
       // Cij_11 = _mm256_add_pd(Cij_11, _mm256_mul_pd(Aik_2,Bkj_3));
      }
      _mm256_store_pd(Cs1, Cij);
      _mm256_store_pd(Cs2, Cij_1);
      _mm256_store_pd(Cs3, Cij_2);
      _mm256_store_pd(Cs4, Cij_3);


      _mm256_store_pd(Cs5, Cij_4);
      _mm256_store_pd(Cs6, Cij_5);
      _mm256_store_pd(Cs7, Cij_6);
      _mm256_store_pd(Cs8, Cij_7);

 //     _mm256_store_pd(C+i+8+(j+3)*lda, Cij_8);
 //     _mm256_store_pd(C+i+8+(j+3)*lda, Cij_9);
 //     _mm256_store_pd(C+i+8+(j+3)*lda, Cij_10);
 //     _mm256_store_pd(C+i+8+(j+3)*lda, Cij_11);
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (const int lda, double*  A, double* B, double* restrict C)
{

  // MODE = 0 -> use 128avx and unroll
  // MODE = 1 -> use 256avx and unroll
 // int MODE = 1;

  int newlda = lda;
  int div = 8;
  if (lda % div){
    int t = lda % div;
    newlda = lda + (div-t);
  }

  double* padA = (double*) _mm_malloc(newlda * newlda * sizeof(double), 32);
  pad(padA, A, lda, newlda);

  double* padB = (double*) _mm_malloc(newlda * newlda * sizeof(double), 32);
  pad(padB, B, lda, newlda);

  double* padC = (double*) _mm_malloc(newlda * newlda * sizeof(double), 32);
  pad(padC, C, lda, newlda);

  /* For each block-row of A */ 
  for (int i = 0; i < newlda; i += BLOCK_SIZE2)
  {
    /* For each block-column of B */
    for (int j = 0; j < newlda; j += BLOCK_SIZE)
    {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < newlda; k += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min (BLOCK_SIZE2, newlda-i);
        int N = min (BLOCK_SIZE, newlda-j);
        int K = min (BLOCK_SIZE, newlda-k);
        /* Perform individual block dgemm */

   //     if (MODE == 0) do_block_unrollavx128(newlda, M, N, K, padA + i + k*newlda, padB + k + j*newlda, padC + i + j*newlda);
       // else 
       do_block_unrollavx256(newlda, M, N, K, padA + i + k*newlda, padB + j*newlda + k, padC + i + j*newlda);

      }
    }
  }

  unpad(padC, C, lda, newlda);
  _mm_free(padA);
  _mm_free(padB);

}
