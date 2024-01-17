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
    long int N, long int offset, long int n_per_rank,
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
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return mismatch_counter;
}

double time_mvm_serial(double ** restrict A, double * restrict x, double * restrict y, 
                       long int N, int niter) 
{
  double runtime = 0.0;  //Measured runtime

  double * xtmp = malloc(N * sizeof(double));
  
  memcpy(xtmp, x, N * sizeof(double));
  /* Measure time only in the internal loop */
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

double time_mvm(double ** restrict A_local, double * restrict x_local, double * restrict y_local, 
                long int rows_per_rank, long int cols_per_rank, long int N, 
                int niter, MPI_Comm mtx_row_comm, MPI_Comm mtx_col_comm) 
{
  int rank, row_color, col_color;
  int row_comm_rank = 0;
  int row_comm_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // UNCOMMENT once mtx_row_comm is properly initialised
  // MPI_Comm_rank(mtx_row_comm, &row_comm_rank);
  // MPI_Comm_size(mtx_row_comm, &row_comm_size);

  row_color = rank / row_comm_size;
  col_color = row_comm_rank;

  double * xtmp = malloc(cols_per_rank * sizeof(double));
  double * ytmp = malloc(rows_per_rank * sizeof(double));
  double * yred = malloc(rows_per_rank * sizeof(double));
  double * yglobal = malloc(N * sizeof(double));

  double runtime = 0.0;  //Measured runtime

  memcpy(xtmp, x_local, cols_per_rank * sizeof(double));
  
  double start = MPI_Wtime();
  
  double tmp;
  for(int iter = 0; iter < niter; ++iter){
    for (long int i = 0; i<rows_per_rank; i++){
      tmp = 0;
      for (long int j = 0; j<cols_per_rank; j++){
        tmp += A_local[i][j] * xtmp[j];
      }
      ytmp[i] = tmp;
    }
    // sum up contributions to ytmp from the ranks reponsible for different sets of columns
    //MPI_Allreduce(ytmp, yred, ...);
    if( niter > 1 ){
      // we again want to compute A*A*...*x, so we have to take the result of the previous
      // iteration as the input of the next application of A
      // we thus need to fetch the right components of the global 'ytmp' 
      // one easy (but wasteful) way to do so is to employ an Allgather
      // MPI_Allgather(yred, ...);
      // now that we have the full y, need to copy the right part to xtmp as input for the next
      // iteration
      // memcpy(xtmp, ...);

      // the alternative would be to get just the elements that we need using
      // MPI_Allgatherv or individual MPI_Send / MPI_Recv
    }
  }
  // copy result over to output vector
  memcpy(y_local, yred, rows_per_rank * sizeof(double));
  
  /* Measure runtime per iteration in seconds */
  runtime = (MPI_Wtime()-start)/niter;

  free(xtmp);
  free(ytmp);
  free(yred);
  free(yglobal);
  return runtime;
}

int main(int argc, char* argv[])
{
  /* Call this program giving the length of the vectors via command line */
 
  /* Define variables */
  long int size = 32;  //size of vector given via command line, default 32
  double a = 0.0;        //constant
  double *x, *x_local;             //vector x
  double *y, *y_local;             //vector y (result)
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

  // number of ranks per row, default 1
  int n_ranks_row = 1;

  /* Collect the size of the vector from command line */
  if(argc > 1)
    size = (long int) atol(argv[1]);
  else
    printf("Missing 1st argument. Default vector size = %ld\n", size);

  // now the input for n_ranks_row
  if(argc > 2)
    n_ranks_row = atoi(argv[2]);
  else
    printf("Missing 2nd argument. Default n_ranks_row = %d\n", n_ranks_row);
  
  mpi_error = MPI_Init(&argc, &argv);
  check_mpi_error(mpi_error, __FILE__, __LINE__-1);

  int rank = 0;
  int n_ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  int n_row_colors = n_ranks / n_ranks_row;

  if( n_ranks % n_ranks_row != 0 ){
    if( rank == 0 ) printf("Total number of ranks %d must be divisible by n_ranks_row %d! Exiting.\n",
                           n_ranks, n_ranks_row);
    mpi_exit(100);
  }

  if( size % n_ranks_row != 0 || size % n_row_colors ){
    if( rank == 0 ) printf("size %ld must be divisible by n_ranks_row (%d) and n_ranks/n_ranks_row (%d)! Exiting.\n", 
                           size, n_ranks_row, n_row_colors);
    mpi_exit(101);
  }
 
  const int row_color = 0; // color of the row communicator for this rank
  const int row_rank = 0; // rank assignment in the row communicator for this rank
  MPI_Comm mtx_row_comm;
  // split MPI_COMM_WORLD rowwise into mtx_row_comm
  // MPI_Comm_split(MPI_COMM_WORLD, ...);

  const long int rows_per_rank = size / n_row_colors;
  const long int cols_per_rank = size / n_ranks_row; 
  const long int row_offset = 0; // offset which specifies the starting row of the submatrix
                                 // held by this rank

  const int col_color = 0; // color of the column communicator for this rank
  const int col_rank = 0; // rank assignment in the column communicator for this rank
  MPI_Comm mtx_col_comm;
  // split MPI_COMM_WORLD columnwise into mtx_col_com
  // MPI_Comm_split(MPI_COMM_WORLD, ...);
  
  const long int col_offset = 0; // offset which specifies the starting column of the submatrix
                                 // held by this rank

  /* Allocate vectors and matrix */
  x = (double*) malloc(size*sizeof(double)); 
  x_local = (double*) malloc(cols_per_rank*sizeof(double));

  y = (double*) malloc(size*sizeof(double));
  y_local = (double*) malloc(rows_per_rank*sizeof(double));

  A = (double **) malloc(size * sizeof(double*));
  A_local = (double **) malloc(rows_per_rank * sizeof(double*));

  Amem = (double *) malloc(size*size * sizeof(double));
  for (long int i = 0; i < size; i++) 
    A[i] = Amem + i*size;
  
  Amem_local = (double *) malloc(rows_per_rank*cols_per_rank * sizeof(double));
  for (long int i = 0; i < rows_per_rank; i++) 
    A_local[i] = Amem_local + i*cols_per_rank;

  /* Initialize A, x, and y */    
  for (long int i=0; i<size; i++)
  {
    x[i] = ((double)i)/size;
    // initialize x_local here
    //
    y[i] = 0.0;
    // initialize y_local here
    // 
    for (long int j=0; j<size; j++)
    {
      A[i][j] = ((double)(i+j)) / (size*size);
      // initialize A_local here
      //
    }
  }
  // make sure that our associations of global to local data are correct
  long int mismatch_counter = compare_global_local(x, x_local, size, col_offset, cols_per_rank, "x");
  if( mismatch_counter > 0 ){
    printf("Rank %d, %ld mismatches in x_local\n", rank, mismatch_counter);
  }
  
  mismatch_counter = compare_global_local(y, y_local, size, row_offset, rows_per_rank, "y");
  if( mismatch_counter > 0 ){
    printf("Rank %d, %ld mismatches in y_local\n", rank, mismatch_counter);
  }
 
  time_mvm_serial(A, x, y, size, 3);
  time_mvm(A_local, x_local, y_local, rows_per_rank, cols_per_rank, size, 3,
           mtx_row_comm, mtx_col_comm);
  mismatch_counter = compare_global_local(y, y_local, size, row_offset, rows_per_rank, "y");
  if( mismatch_counter == 0 ){
    printf("Rank %d: no mismatches\n", rank);
  } else {
    printf("Rank %d, %ld mismatches in result\n", rank, mismatch_counter);
  }

  // run a few times to fill the caches
  t_iter = time_mvm(A_local, x_local, y_local, 
                    rows_per_rank, cols_per_rank, size, 3, mtx_row_comm, mtx_col_comm);
 
  /* Run the dense repeated matrix-vector multiply function for 20 iterations 
   * Repeat this 5 times and collect the best and worst values
   */
  for (int it=0; it<5; it++)
  {
      t_iter = time_mvm(A_local, x_local, y_local, 
                        rows_per_rank, cols_per_rank, size, 20, mtx_row_comm, mtx_col_comm);
      if (t_best > t_iter) t_best = t_iter;
      if (t_worst < t_iter) t_worst = t_iter; 
      
      /* Reinitialize y */
      for (long int i=0; i<rows_per_rank; i++)
          y[i] = 0.0;
  }
  
  /* Calculate the runtime as the best time, take also the minumum over all MPI ranks */
  // MPI_Allreduce(&t_best, ...);



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
  
  free (x);
  free (y);
  free (A);
  free (Amem);
  free (x_local);
  free (y_local);
  free (A_local);
  free (Amem_local);

  MPI_Finalize();
  return 0;
}

