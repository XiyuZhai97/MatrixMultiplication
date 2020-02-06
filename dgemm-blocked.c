#include <immintrin.h>

const char* dgemm_desc = "Simple blocked dgemm";
#pragma GCC optimize("O3","Ofast","inline","unroll-loops","prefetch-loop-arrays")

#if !defined(BLOCK_SIZE_M)
#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_N 64
#define BLOCK_SIZE_K 64
#endif

#define min(a,b) (((a)<(b))?(a):(b))

static void pad(double* restrict A_padded, double* restrict A, const int lda, const int mat_row)
{
  for (int j = 0; j < lda; ++j){
    for (int i = 0; i < lda; ++i){
      A_padded[i + j * mat_row] = A[i + j * lda];
    }
  }
}

static void unpad(double* restrict A_padded, double* restrict A, const int lda, const int mat_row)
{
  for (int j = 0; j < lda; ++j){
    for (int i = 0; i < lda; ++i){
      A[i + j * lda] = A_padded[i + j * mat_row];
    }   
  }   
}

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static inline void my_do_block(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C)
{
  __m256d Aik, Aik_1, Bkj, Bkj_1, Bkj_2, Bkj_3;
  __m256d Cij_0, Cij_1, Cij_2, Cij_3;
  __m256d Cij_4, Cij_5, Cij_6, Cij_7;
  double *Bs,*Cs1,*Cs2,*Cs3,*Cs4,*Cs5,*Cs6,*Cs7,*Cs8,*As;
  for (int i = 0; i < M; i +=8 ) {
    for (int j = 0; j < N; j += 4) {
      Cs1 = C + i + j * lda;
      Cij_0 = _mm256_load_pd(Cs1);
      Cs2 = Cs1 + lda;
      Cij_1 = _mm256_load_pd(Cs2);
      Cs3 = Cs2 + lda;
      Cij_2 = _mm256_load_pd(Cs3);
      Cs4 = Cs3 + lda;
      Cij_3 = _mm256_load_pd(Cs4);
      Cs5 = Cs1 + 4;
      Cij_4 = _mm256_load_pd(Cs5);
      Cs6 = Cs5 + lda;
      Cij_5 = _mm256_load_pd(Cs6);
      Cs7 = Cs6 + lda;
      Cij_6 = _mm256_load_pd(Cs7);
      Cs8 = Cs7 + lda;
      Cij_7 = _mm256_load_pd(Cs8);

      for (int k = 0; k < K; ++k) {
        As=A+i+k*lda;
        Aik = _mm256_load_pd(As);
	      As += 4;
        Aik_1 = _mm256_load_pd(As);
        Bs = B + k + j*lda;
        Bkj = _mm256_broadcast_sd(Bs);
	      Bs += lda;
        Bkj_1 = _mm256_broadcast_sd(Bs);
	      Bs += lda;
        Bkj_2 = _mm256_broadcast_sd(Bs);
	      Bs += lda;
        Bkj_3 = _mm256_broadcast_sd(Bs);

        Cij_0 = _mm256_add_pd(Cij_0, _mm256_mul_pd(Aik,Bkj));
        Cij_1 = _mm256_add_pd(Cij_1, _mm256_mul_pd(Aik,Bkj_1));
        Cij_2 = _mm256_add_pd(Cij_2, _mm256_mul_pd(Aik,Bkj_2));
        Cij_3 = _mm256_add_pd(Cij_3, _mm256_mul_pd(Aik,Bkj_3));

        Cij_4 = _mm256_add_pd(Cij_4, _mm256_mul_pd(Aik_1,Bkj));
        Cij_5 = _mm256_add_pd(Cij_5, _mm256_mul_pd(Aik_1,Bkj_1));
        Cij_6 = _mm256_add_pd(Cij_6, _mm256_mul_pd(Aik_1,Bkj_2));
        Cij_7 = _mm256_add_pd(Cij_7, _mm256_mul_pd(Aik_1,Bkj_3));
      }
        _mm256_store_pd(Cs1, Cij_0);
        _mm256_store_pd(Cs2, Cij_1);
        _mm256_store_pd(Cs3, Cij_2);
        _mm256_store_pd(Cs4, Cij_3);

        _mm256_store_pd(Cs5, Cij_4);
        _mm256_store_pd(Cs6, Cij_5);
        _mm256_store_pd(Cs7, Cij_6);
        _mm256_store_pd(Cs8, Cij_7);

    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm(const int lda, double* A, double* B, double* restrict C)
{
  int mat_row = lda;
  int mat_col = lda;
  if (lda % 8){
    int t = lda % 8;
    mat_row = lda + (8 - t);
    mat_col = lda + (8 - t);
  }

  //block size now is mat_row*mat_col
  double* A_padded = (double*) _mm_malloc(mat_row * mat_col * sizeof(double), 64);
  double* B_padded = (double*) _mm_malloc(mat_row * mat_col * sizeof(double), 64);
  double* C_padded = (double*) _mm_malloc(mat_row * mat_col * sizeof(double), 64);
  pad(A_padded, A, lda, mat_row);
  pad(B_padded, B, lda, mat_row);
  pad(C_padded, C, lda, mat_row);
  for (int i = 0; i < mat_row; i += BLOCK_SIZE_M){
    for (int j = 0; j < mat_row; j += BLOCK_SIZE_N){
      for (int k = 0; k < mat_row; k += BLOCK_SIZE_K){
        int M = min (BLOCK_SIZE_M, mat_row-i);
        int N = min (BLOCK_SIZE_N, mat_row-j);
        int K = min (BLOCK_SIZE_K, mat_row-k);
        my_do_block(mat_row, M, N, K, A_padded + i + k*mat_row, B_padded + j*mat_row + k, C_padded + i + j*mat_row);
      }
    }
  }
  unpad(C_padded, C, lda, mat_row);
  _mm_free(A_padded);
  _mm_free(B_padded);
}
