#include <math.h>
#include <stdio.h>
const char *dgemm_desc = "Simple unroll blocked dgemm.";
#pragma GCC optimize(3,"Ofast","inline")

#define min(a, b) (((a) < (b)) ? (a) : (b))

static void my_do_block( int lda, int M, int N, int K, double *A, double *B, double *C )
{
  for( int i = 0; i < M; i++ )
       for( int j = 0; j < N; j++ ) 
       {
            double cij = C[i+j*lda];
            double register temp0, temp1,temp2, temp3;
            double register a0, a1, a2, a3;
            double register b0, b1, b2, b3;

            int k=0;
            for( k = 0; k < K-3; k+=4 )
            {
                 a0=A[i+k*lda];
                 a1=A[i+(k+1)*lda];
                 a2=A[i+(k+2)*lda];
                 a3=A[i+(k+3)*lda];

                 b0=B[k+j*lda];
                 b1=B[k+j*lda+1];
                 b2=B[k+j*lda+2];
                 b3=B[k+j*lda+3];
                          
                 temp0=a0 * b0;
                 temp1=a1 * b1;
                 temp2=a2 * b2;
                 temp3=a3 * b3;

                 cij +=temp0+temp1+temp2+temp3;
            }
            for( int h=k;h < K; h++) // missing elements
                 cij += A[i+h*lda] * B[h+j*lda];
            C[i+j*lda] = cij;
       }
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C) {
    // For each block-row of A
    int BLOCK_SIZE = 32;
    int miniblock_M = 4;
    int miniblock_N = 2;
    int miniblock_K = 2;
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