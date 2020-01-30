#include <x86intrin.h>
#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
const char *dgemm_desc = "Simple blocked dgemm.";
#pragma GCC optimize(3,"Ofast","inline")
// #pragma pack(16)
#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
// static void do_block(int lda, int M, int N, int K, double *A, double *B, double *C) {
//     // For each row i of A

//     for (int i = 0; i < M; ++i) {
//         //For each column j of B
//         for (int j = 0; j < N; ++j) {
//             // Compute C(i,j)
//             double cij = C[i + j * lda];
//             for (int k = 0; k < K; ++k) {
//                 cij += A[i + k * lda] * B[k + j * lda];
//             }
//             C[i + j * lda] = cij;
//         }
//     }
// }
static void my_do_block( int lda, int M, int N, int K, double *A, double *B, double *C )
{
  for( int i = 0; i < M; i++ )
  {
       for( int j = 0; j < N; j++ ) 
       {
            __m256d cij =_mm256_set1_pd(C[i+j*lda]);
            __m256d temp0, temp1,temp2, temp3;
            __m256d a0, a1, a2, a3;
            __m256d b0, b1, b2, b3;
            int k = 0;
            for(k = 0; k < K-3; k+=4 )
            {
                a0=_mm256_set1_pd(A[i+k*lda]);
                a1=_mm256_set1_pd(A[i+(k+1)*lda]);
                a2=_mm256_set1_pd(A[i+(k+2)*lda]);
                a3=_mm256_set1_pd(A[i+(k+3)*lda]);

                b0=_mm256_set1_pd(B[k+j*lda]);
                b1=_mm256_set1_pd(B[k+j*lda+1]);
                b2=_mm256_set1_pd(B[k+j*lda+2]);
                b3=_mm256_set1_pd(B[k+j*lda+3]);

                temp0=_mm256_mul_pd(a0 , b0);
                temp1=_mm256_mul_pd(a1 , b1);
                temp2=_mm256_mul_pd(a2 , b2);
                temp3=_mm256_mul_pd(a3 , b3);

                cij =_mm256_add_pd(cij,temp0);
                cij =_mm256_add_pd(cij,temp1);
                cij =_mm256_add_pd(cij,temp2);
                cij =_mm256_add_pd(cij,temp3);
            }
            for( int h = k; h < K; h++){ // missing elements
                a0=_mm256_set1_pd(A[i + h*lda]);
                b0=_mm256_set1_pd(B[h + j*lda]);
                temp0=_mm256_mul_pd(a0 , b0);
                cij =_mm256_add_pd(cij,temp0);
            }
            // double *cij_temp;
            _mm256_store_pd(C + i + j*lda , cij);
            // C[i+j*lda] = cij_temp;
       }
    }
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C) {
    A = (double *)_mm_malloc(lda * sizeof(double), 32);
    B = (double *)_mm_malloc(lda * sizeof(double), 32);
    C = (double *)_mm_malloc(lda * sizeof(double), 32);


    // For each block-row of A
    int BLOCK_SIZE = 32;
    int miniblock_M = 4;
    int miniblock_N = 4;
    int miniblock_K = 4;
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                for (int ii = 0; ii < M; ii += miniblock_M) {
                    // For each block-column of B
                    for (int jj = 0; jj < N; jj += miniblock_N) {
                        // Accumulate block dgemms into block of C
                        for (int kk = 0; kk < K; kk += miniblock_K) {
                            int MM=min(miniblock_M,M-ii);
                            int NN=min(miniblock_N,N-jj);
                            int KK=min(miniblock_K,K-kk);
                            my_do_block(lda, MM, NN, KK, A + i +ii + (k+kk) * lda, B + k+kk + (j+jj) * lda, C + i+ii + (j+jj) * lda);
                        }
                    }
                }
            }
        }
    }
}