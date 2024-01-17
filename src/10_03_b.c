/* Dense Matrix-vector Multiplication
 * E.Suarez (FZJ/UBonn, 2023)
 * B.Kostrzewa (UBonn, 2023)
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* For LONG_MAX */
#include <limits.h>

#include <sys/types.h>
#include <unistd.h>

void mpi_exit(const int error){
  int inited = 0;
  MPI_Initialized(&inited);
  if(inited) MPI_Finalize();
  exit(error);    
}

void check_mpi_error(const int error, const char * filename, const int line)
{
  if( error != MPI_SUCCESS )
  {    
    printf("MPI error %d at %s, line %d\n", error, filename, line);
    mpi_exit(error);
  }    
}

long int compare_global_local(
    double * x, double * x_local, 
    long int offset, long int n_per_rank,
    const char * name){
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  long int mismatch_counter = 0;
  for(long int i = offset; i < offset + n_per_rank; ++i){
    if( fabs(x[i] - x_local[i-offset]) > 2*DBL_EPSILON ){
      printf("Mismatch: rank %d %s[%ld] %s_local[%ld]: %e %e\n", 
             rank, name, i, name, i-offset, x[i], x_local[i-offset]);
      mismatch_counter++;
    }
  }
  return mismatch_counter;
}

long int compare_vec(double * y, double * y_ref, long int N){
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  long int mismatch_counter = 0;
  for(long int i = 0; i < N; ++i){
    if( fabs(y[i] - y_ref[i]) > 2*DBL_EPSILON ){
      printf("Mismatch: rank %d y[%ld] y_ref[%ld]: %e %e\n", 
             rank, i, i, y[i], y_ref[i]);
      mismatch_counter++;
    }
  }
  return mismatch_counter;
}

double time_mvm_serial(double ** restrict A, double * restrict x, double * restrict y, 
                       long int N, int niter) 
{
  double runtime = 0.0;  //Measured runtime

  double * xtmp = malloc(N * sizeof(double));
  
  memcpy(xtmp, x, N * sizeof(double));

  double start = MPI_Wtime();
  
  double tmp;
  for(int iter = 0; iter < niter; ++iter){
    for (long int i = 0; i<N; i++){
      tmp = 0;
      for (long int j = 0; j<N; j++){
        tmp += A[i][j] * xtmp[j];
      }
      y[i] = tmp;
    }
    // copy output to 'xtmp' for the next iteration 
    if(niter > 1) memcpy(xtmp, y, N * sizeof(double));
  }
  
  /* Measure runtime per iteration in seconds */
  runtime = (MPI_Wtime()-start)/niter;
  free(xtmp);
  return runtime;
}

double time_mvm(double ** restrict A_local, double * restrict x_local, double * restrict y, 
                long int N, long int cols_per_rank, long int niter) 
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double * xtmp = malloc(cols_per_rank * sizeof(double));
  double * ytmp = malloc(N * sizeof(double));
  double * yred = malloc(N * sizeof(double));

  double runtime = 0.0;  //Measured runtime

  memcpy(xtmp, x_local, cols_per_rank * sizeof(double));

  /* Measure time only in the internal loop */
  double start = MPI_Wtime();
  
  double tmp;
  for(int iter = 0; iter < niter; ++iter){
    for (long int i = 0; i<N; i++){
      tmp = 0;
      for (long int j = 0; j<cols_per_rank; j++){
        tmp += A_local[i][j] * xtmp[j];
      }
      ytmp[i] = tmp;
    }
    // sum up ytmp from all ranks into yred
    //MPI_Allreduce(ytmp, ...);
    if( niter > 1 ){
      // for next iteration, extract the right part of yred into xtmp
      //memcpy(xtmp, ...);
    }
  }
  // copy the final result into the output
  memcpy(y, yred, N * sizeof(double));
  
  /* Measure runtime per iteration in seconds */
  runtime = (MPI_Wtime()-start)/niter;
  free(xtmp);
  free(ytmp);
  free(yred);
  return runtime;
}

int main(int argc, char* argv[])
{
  /* Call this program giving the length of the vectors via command line */
 
  /* Define variables */
  long int size = 32;  //size of vector given via command line, default 32
  double a = 0.0;        //constant
  double *x, *x_local;             //vector x
  double *y, *y_serial;             //vector y (result)
  double **A, **A_local;            //matrix A
  double *Amem, *Amem_local;          //matrix elements of A
  double t_iter = 0.0;    //runtime [sec]
  double t_best = 1000.0;   //best timing [sec]
  double t_worst = 0.0;   //worst timing [sec]
  
  long int flops = 0.0;  //floating point operations executed
  long int kernel_bytes = 0.0;  //bytes  transfered
  long int working_set = 0.0; // working set size (in bytes)
  double kernel_bw = 0.0;     //memory bandwidth    
  double mem_bw = 0.0;     //memory bandwidth    
  double perf = 0.0;   //performance (flops/s)

  int mpi_error;
  int n_ranks_row = 1;
  int ny_ranks = 1;

  /* Collect the size of the vector from command line */
  if(argc > 1)
    size = (long int) atol(argv[1]);
  else
    printf("Missing 1st argument. Default vector size = %ld\n", size);

  mpi_error = MPI_Init(&argc, &argv);
  check_mpi_error(mpi_error, __FILE__, __LINE__-1);

  int rank = 0;
  int n_ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  if( size % n_ranks != 0 ){
    if( rank == 0 ) printf("size %ld must be divisible by n_ranks %d! Exiting.\n", 
                           size, n_ranks);
    mpi_exit(100);
  }
 
  const long int cols_per_rank = size / n_ranks;
  const long int col_offset = rank * cols_per_rank;

  /* Allocate vectors and matrix */
  x = (double*) malloc(size*sizeof(double)); 
  x_local = (double*) malloc(cols_per_rank*sizeof(double));

  y = (double*) malloc(size*sizeof(double));
  y_serial = (double*) malloc(size*sizeof(double));

  A = (double **) malloc(size * sizeof(double*));
  A_local = (double **) malloc(size * sizeof(double*));

  Amem = (double *) malloc(size*size * sizeof(double));
  for (long int i = 0; i < size; i++) 
    A[i] = Amem + i*size;
  
  Amem_local = (double *) malloc(size*cols_per_rank * sizeof(double));
  for (long int i = 0; i < size; i++) 
    A_local[i] = Amem_local + i*cols_per_rank;

  /* Initialize A, x, and y */    
  for (long int i=0; i<size; i++)
  {
    x[i] = ((double)i)/size;
    // initialise x_local here
    //
    y[i] = 0.0;
    y_serial[i] = y[i];
    for (long int j=0; j<size; j++)
    {
      A[i][j] = ((double)(i+j)) / (size*size);
      // initialize A_loccal here
      //
    }
  }
  // make sure that x_local is initialised properly
  long int mismatch_counter = compare_global_local(x, x_local, col_offset, cols_per_rank, "x");
  if( mismatch_counter > 0 ){
    printf("Rank %d: %ld mismatches in x_local!\n", rank, mismatch_counter);
  }
  
  // test if your parallel implementation works correctly
  time_mvm_serial(A, x, y_serial, size, 3);
  time_mvm(A_local, x_local, y, size, cols_per_rank, 3);
  mismatch_counter = compare_vec(y, y_serial, size);
  if( mismatch_counter == 0 ){
    printf("Rank %d: no mismatches\n", rank);
  }

  // run a few times to fill the caches
  t_iter = time_mvm(A_local, x_local, y, size, cols_per_rank, 3);
 
  /* Run the dense repeated matrix-vector multiply function for 20 iterations 
   * Repeat this 5 times and collect the best and worst values
   */
  for (int it=0; it<5; it++)
  {
      t_iter = time_mvm(A_local, x_local, y, size, cols_per_rank, 20);
      if (t_best > t_iter) t_best = t_iter;
      if (t_worst < t_iter) t_worst = t_iter; 
  }
  
  /* Calculate the runtime as the best time, take also the minumum over all MPI ranks */
  //MPI_Allreduce(&t_best, &t_iter, ...);
  
  /* To calculate each element of vector "y":
   * BASED ON LOOP:
   * for (long int i = 0; i<N; i++)
   *       for (long int j = 0; j<N; j++)
   *           y[i] = y[i] + A[i][j] * x[j];
   * OPERATIONS
   *     - Innerloop: N* (1 Sum + 1 Muliply) = 2*N
   *     - Outerloop: N*Innerloop = 2*N*N
   * MEMORY ACCESSES
   *     - Innerloop: N* (3 loads (y[j], A[i][j]), x[j]) + 1 store y[j]) =
   *                = 4*N
   *     - Outerloop: N*Innerloop = 4*N*N (*Factor 8 Byte for DP)
   */
  
  flops = 2*size*size;
  kernel_bytes = 8*4*size*size;
  working_set = 8*(size*size + 2*size); // matrix A + vectors x and y
  perf = (double) flops / t_iter;   
  kernel_bw = (double) kernel_bytes / t_iter; 
  mem_bw = (double) working_set / t_iter;
  
  /* Size FLOP Bytes kernel-BW[GB/s] working_set mem-BW[GB/s] Runtime[ms] */
  if( rank == 0 ){
    printf("%5d %10ld  %10ld  %10.2e  %10.2e  %11.2e  %10.2e  %10.3e\n", 
           n_ranks, size, flops, (double)kernel_bytes/1.e9, kernel_bw/1.e9, (double)working_set/1.e9, mem_bw/1.e9, t_iter*1e3);
  }
  
  free(x);
  free(y);
  free(A);
  free(Amem);
  free(y_serial);
  free(x_local);
  free(A_local);
  free(Amem_local);

  MPI_Finalize();
  return 0;
}

