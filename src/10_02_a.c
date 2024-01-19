#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

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

long int compare_accum_vec(double * x, double * y, long int N){
  int rank = 0;
  int n_ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  long int mismatch_counter = 0;

  for(long int i = 0; i < N; ++i){
    // we use a rather relaxed criterion as we know that different
    // ways of performing the reduction will cause rounding errors
    // which may show up here otherwise
    if( fabs(x[i] - y[i]) > 10*n_ranks*DBL_EPSILON ){
      printf("Mismatch: rank %d x[%ld] y[%ld]: %e %e\n",
             rank, i, i, x[i], y[i]);
      mismatch_counter++;
    }
  }
  return mismatch_counter;
}


int main(int argc, char ** argv)
{
  // in principle almost every MPI function returns an integer to communicate
  // the success or failure of the operation
  int mpi_error;
  mpi_error = MPI_Init(&argc, &argv);
  check_mpi_error(mpi_error, __FILE__, __LINE__-1);

  int rank = 0;
  int n_ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  const long int vec_size = 32;

  double * local_vec = (double*) malloc(sizeof(double) * vec_size);
  double * reduced_vec = (rank == 0) ? (double*) malloc(sizeof(double) * vec_size) : NULL;
  double * remote_vec = (rank == 0) ? (double*) malloc(sizeof(double) * vec_size) : NULL;
  double * accum_vec = (rank == 0) ? (double*) malloc(sizeof(double) * vec_size) : NULL;

  srand(rank + 1234);
  for(long int i = 0; i < vec_size; ++i){
    local_vec[i] = ((double)rand())/RAND_MAX;
    if(rank == 0) accum_vec[i] = local_vec[i];
  }
  
  // perform an element-wise sum of the local_vec (residing on all ranks), summing into
  // 'reduced_vec' on rank 0
  MPI_Reduce(local_vec, reduced_vec, vec_size, MPI_DOUBLE, MPI_SUM, 0,
           MPI_COMM_WORLD);

  for(int r = 1; r < n_ranks; ++r){
    if(rank == 0){
      MPI_Recv(remote_vec, vec_size, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for(long int i = 0; i < vec_size; ++i){
        accum_vec[i] += remote_vec[i];
      }
    } else {
      MPI_Send(local_vec, vec_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
  }
  
  long int mismatch_counter = 0;
  if( rank == 0 ){
    mismatch_counter = compare_accum_vec(reduced_vec, accum_vec, vec_size);
    if( mismatch_counter == 0 ){
      printf("No mismatches.\n");
    }
  }
  
  // use MPI_Bcast to communicate the value of mismatch_counter from rank 0 to all other ranks
  // MPI_Bcast(&mismatch_counter, 1, MPI_LONG, 0, MPI_COMM_WORLD);

  free(local_vec);
  if(rank == 0){
    free(reduced_vec);
    free(remote_vec);
    free(accum_vec);
  }
  MPI_Finalize();
  return mismatch_counter > 0 ? 100 : 0;
}
