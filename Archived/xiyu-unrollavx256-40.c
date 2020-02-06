#include <immintrin.h>
#include "string.h"
#include <stdio.h>

const char* dgemm_desc = "Blocked dgemm with padding and AVX.";

#if !defined(BLOCK_SIZEk)
#define BLOCK_SIZEk 64
#define BLOCK_SIZEj 64
#define BLOCK_SIZEi 32
#endif

#define min(a,b) (((a)<(b))?(a):(b))

///copy array into the padded matrix
static void pad(double* restrict padA, double* restrict A, const int lda, const int matsize)
{
  for (int j = 0; j < lda; j++) {
    for (int i = 0; i < lda; i++) {
      padA[i + j*matsize] = A[i + j*lda];
    }
  }
}

//copy array into the unpadded matrix
static void unpad(double* restrict padA, double* restrict A, const int lda, const int matsize)
{
  for (int j = 0; j < lda; j++) {
    for (int i = 0; i < lda; i++) {
      A[i + j*lda] = padA[i + j*matsize];
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
static void do_block_unrollavx256(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C)
{
  __m256d Aik, Aik_1, Bkj, Bkj_1, Bkj_2, Bkj_3;
  __m256d Cij, Cij_1, Cij_2, Cij_3, Cij_4, Cij_5, Cij_6, Cij_7;

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

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (const int lda, double* restrict A, double* restrict B, double* restrict C)
{
    int matsize = lda;
    int div = 8;
    int rem = lda % div;
    if (rem){
        matsize = lda + (div-rem) + div;
    }

    double* padA = (double*) _mm_malloc(matsize * matsize * sizeof(double), 32);
    double* padB = (double*) _mm_malloc(matsize * matsize * sizeof(double), 32);
    double* padC = (double*) _mm_malloc(matsize * matsize * sizeof(double), 32);
    pad(padA, A, lda, matsize);
    pad(padB, B, lda, matsize);
    pad(padC, C, lda, matsize);

    for (int k = 0; k < matsize; k += BLOCK_SIZEk)
    {
      for (int j = 0; j < matsize; j += BLOCK_SIZEj)
      {
        for (int i = 0; i < matsize; i += BLOCK_SIZEi)
        {
          int M = min (BLOCK_SIZEi, matsize-i);
          int N = min (BLOCK_SIZEj, matsize-j);
          int K = min (BLOCK_SIZEk, matsize-k);
          do_block_unrollavx256(matsize, M, N, K, padA + i + k*matsize, padB + k + j*matsize, padC + i + j*matsize);
        }
      }
    }

    unpad(padC, C, lda, matsize);
    _mm_free(padA);
    _mm_free(padB);
}
