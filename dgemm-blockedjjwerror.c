#include <math.h>
#include <stdio.h>
#include <stdlib.h>
const char *dgemm_desc = "Simple blocked dgemm.";

//#ifndef BLOCK_SIZE
//#define BLOCK_SIZE 64
//#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
 //__attribute__((optimize("unroll-loops")))
inline static void do_block(int lda,int M, int N, int K, int MM, int NN, int KK, double *A, double *B, double* C) {
    // For each row i of A

    for (int i = 0; i < MM; ++i) {
        //For each column j of B
        for (int j = 0; j < NN; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (int k = 0; k < KK; ++k) {
                cij += A[i + k * K] * B[k + j * N];
		printf("i:%d,j:%d,k:%d,MM:%d,NN:%d,KK:%d\n",i,j,k,MM,NN,KK);
            }
            C[i + j * lda] = cij;
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
// __attribute__((optimize("unroll-loops")))
void square_dgemm(int lda, double *A, double *B, double * restrict C) {
    // For each block-row of A
    int row = 22;
    int col = 22;
    int inn = 22;
    int minirow = 2;
    int minicol = 2;
    int miniinn = 2;
    for (int i = 0; i < lda; i += row) {
        // For each block-column of B
        for (int j = 0; j < lda; j += col) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += inn) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(row, lda - i);
                int N = min(col, lda - j);
                int K = min(inn, lda - k);
		double * Abblock = (double *) malloc(M*K* sizeof(double));
		double * Bbblock = (double *) malloc(K*N*sizeof(double));
		printf("define A,B BLOCK,M:%d,N:%d,K:%d\n",M,N,K);
//		double Cbblock[M*N]__attribute__((aligned(64)));
		//copy A,B,C block to Abblock,Bbblock,Cbblock
	//	for (int cbi = 0;cbi<M;cbi++){
	//		for (int cbj=0;cbj<N;cbj++){
	//		Cbblock[cbi+cbj*N]=C[i+cbi+(j+cbj)*lda];
	//		}
	//	}
		for (int abi = 0;abi<M;++abi){
			for (int abj=0;abj<K;++abj){
			Abblock[abi+abj*K]=A[i+abi+(k+abj)*lda];
			}
		}
		for (int bbi=0;bbi<K;++bbi){
			for (int bbj=0;bbj<N;++bbj){
				Bbblock[bbi+bbj*N]=B[k+bbi+(j+bbj)*lda];
			}
		}
		printf("block size:%d,%d,%d\n",M,N,K);
		//from now on let's use the new aligned variables
    		for (int ii = 0; ii < M; ii += minirow) {
        		// For each block-column of B
        		for (int jj = 0; jj < N; jj += minicol) {
            		// Accumulate block dgemms into block of C
           			for (int kk = 0; kk < K; kk += miniinn) {
					int MM=min(minirow,M-ii);
					int NN=min(minicol,N-jj);
					int KK=min(miniinn,K-kk);
                			do_block(lda,M,N,K, MM, NN, KK, Abblock +ii + kk * K, Bbblock+kk + jj * N, C+i+ii+(j+jj)*lda);
				}
			}
		}
		free(Abblock);
		free(Bbblock);
            }
        }
    }
}
