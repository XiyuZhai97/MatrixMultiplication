#include <x86intrin.h>
#include <stdio.h>
#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
const char *dgemm_desc = "Simple blocked dgemm.";
#pragma GCC optimize(3,"Ofast","inline","unroll-loops","prefetch-loop-arrays")

// BLOCK_L1 AND BLOCK_L2 MUST BE MULTIPLE OF 4
#define BLOCK_L1 256
#define BLOCK_L2 512

#define ARRAY(A,i,j) (A)[(j)*lda + (i)]
#define min(a,b) (((a)<(b))?(a):(b))

static inline void matmul_4xkxkx4(int lda, int K, double* a, double* b, double* restrict c,int rM,int rN)
{
  __m256d a_coli, bi0, bi1, bi2, bi3;
  __m256d c_col0,c_col1,c_col2,c_col3;
  double *c_ptr1,*c_ptr2,*c_ptr3;

  /* layout of 4x4 c matrix
      00 01 02 03
      10 11 12 13
      20 21 22 23
      30 31 32 33
  */
  // load old value of c
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
//static inline void dp(int lda,int resi,int resj,double*a,double*b,double*C){
//  __mm64d clist[4];
//  _mm256_load_pd(a);
//  _mm256_load_pd(b);
//  _mm256_add_pd(clist,a,b);
//  for (int i=0:i<4;i++){
//    cij+=clist[i];
 // }
//  C[resi+resj*lda]=cij;
//}

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
// this function assumes data is stored in col-major
// if data is in row major, call it like matmul4x4(B, A, C)
//void matmul4x4(double *A, double *B, double *C) {
 //   __m256d col[4], sum[4];
  //  //load every column into registers
   // for(int i=0; i<4; i++)  
 //     col[i] = _mm256_load_pd(&A[i*4]);
  //  for(int i=0; i<4; i++) {
//        sum[i] = _mm256_setzero_pd();      
 //       for(int j=0; j<4; j++) {
  //          sum[i] = _mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(B[i*4+j]), col[j]), sum[i]);
  //      }           
   // }
  //  for(int i=0; i<4; i++) 
  //    _mm256_store_pd(&C[i*4], sum[i]); 
//}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block (int lda, int M, int N, int K,int newM,int newN, double* A, double* B, double*restrict C, double* A_block, double* B_block)
{
  //double A_block[M*K], B_block[K*N];
  //double *A_block, *B_block;
  double *a_ptr, *b_ptr;

  /* For each column of B */
  for (int j = 0 ; j < newN; j += 4) 
  {
    b_ptr = &B_block[K * j];
    if (j>N-4) {
      copyadd0_b(lda,K,B+j*lda,b_ptr,N-j);
    }
    else{
    // copy and transpose B_block
    copy_b(lda, K, B + j*lda, b_ptr);
    }
    /* For each row of A */
    for (int i = 0; i < newM; i += 4) {
      a_ptr = &A_block[i * K];
      if (j == 0){
        if (i>M-4){
	  copyadd0_a(lda,K,A+i,a_ptr,M-i);
	}
	else{
        copy_a(lda, K, A + i, a_ptr);
	}
      }
      int rM=min(4,M-i);
      int rN=min(4,N-j);
      matmul_4xkxkx4(lda, K, a_ptr, b_ptr, C+i+j*lda,rM,rN);
    }
  }
  //now do block product of A[resrow,K]*B
//  if (resrow != 0) 
//  {
//    /* For each row of A */
//    for ( ; i < M; ++i)
//      /* For each column of B */ 
//      for (int p = 0; p < N; ++p) 
//      {
//        /* Compute C[i,j] */
//        double c_ip = ARRAY(C,i,p);
//        for (int k = 0; k < K; k++)
//          c_ip += ARRAY(A,i,k) * ARRAY(B,k,p);
//        ARRAY(C,i,p) = c_ip;
//      }
//  }
//  if (rescol != 0) 
//  {
//    /* For each column of B */
//    for ( ; j < N; ++j)
//      /* For each row of A */ 
//      for (i = 0; i < M_floor; ++i) 
//      {
//        /* Compute C[i,j] */
//        double cij = ARRAY(C,i,j);
//        for (int k = 0; k < K; ++k)
//          cij += ARRAY(A,i,k) * ARRAY(B,k,j);
//        ARRAY(C,i,j) = cij;
//      }
//  }
}

// void print_matrix(double *M, int n_row, int n_col) {
//   for (int i = 0; i < n_row; i++){
//     for (int j = 0; j < n_col; j++){
//       printf("%f ", M[i + j * n_row]);
//     }
//     printf("\n");
//   }
//   printf("\n");
// }

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  

 /*
          s
  _________________
  |       ------- | 
  |       |  |  | |
  |      i------- |  
t |       |  |  | |
  |       ------- |
  |          j    | 
  |               |
  |_______________| 
  loop r and k are for accumulation
 */

 void square_dgemm (int lda, double* A, double* B, double*restrict C)
{

  double *A_block, *B_block;
  posix_memalign((void **)&A_block, 64, BLOCK_L1 * BLOCK_L1 * sizeof(double));
  posix_memalign((void **)&B_block, 64, BLOCK_L1 * BLOCK_L1 * sizeof(double));


  // reorder loops for cache efficiency
  for (int t = 0; t < lda; t += BLOCK_L2) 
  {
    // For each block column of B 
    for (int s = 0; s < lda; s += BLOCK_L2) 
    {
      // For each block row of A 
      for (int r = 0; r < lda; r += BLOCK_L2) 
      {
        // compute end index of smaller block
        int end_k = t + min(BLOCK_L2, lda-t);
        int end_j = s + min(BLOCK_L2, lda-s);
        int end_i = r + min(BLOCK_L2, lda-r);

        for (int k = t; k < end_k; k += BLOCK_L1) 
        {
          // For each block column of B 
          for (int j = s; j < end_j; j += BLOCK_L1) 
          {
            // For each block row of A
            for (int i = r; i < end_i; i += BLOCK_L1) 
            {
              // compute L1 block size
              int K = min(BLOCK_L1, end_k-k);
              int N = min(BLOCK_L1, end_j-j);
              int M = min(BLOCK_L1, end_i-i);
//	      printf("M,N,K:%d,%d,%d\n",M,N,K);
//	      fflush( stdout );
              /* Performs a smaller dgemm operation
               *  C' := C' + A' * B'
               * where C' is M-by-N, A' is M-by-K, and B' is K-by-N. */
              //do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);


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
