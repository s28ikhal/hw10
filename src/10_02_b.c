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
  double * reduced_vec = (double*) malloc(sizeof(double) * vec_size);
  double * remote_vec = (double*) malloc(sizeof(double) * vec_size);
  double * accum_vec = (double*) malloc(sizeof(double) * vec_size);

  srand(rank + 1234);
  for(long int i = 0; i < vec_size; ++i){
    local_vec[i] = ((double)rand())/RAND_MAX;
    accum_vec[i] = local_vec[i];
  }
  
  // element-wise accumulation of local_vec from all ranks into reduced_vec
  // on all ranks
  // MPI_Allreduce(local_vec, reduced_vec, ...);

  // step-by-step element-wise accumulation of local_vec from all ranks into reduced_vec
  // on all ranks
  for(int r1 = 0; r1 < n_ranks; ++r1){
    for(int r2 = 0; r2 < n_ranks; ++r2){
      if(r1 == r2){
        continue;
      } else {
        if(r1 == rank){
          //MPI_Recv(remote_vec, ...);
          for(long int i = 0; i < vec_size; ++i){
            accum_vec[i] += remote_vec[i];
          }
        } else if ( r2 == rank ){
          //MPI_Send(local_vec, ...);
        }
      }
    }
  }
  long int mismatch_counter = compare_accum_vec(reduced_vec, accum_vec, vec_size);

  if( mismatch_counter == 0 ){
    printf("Rank %d: no mismatches\n", rank);
  } else {
    printf("Rank %d, %ld mismatches in result\n", rank, mismatch_counter);
  }

  free(local_vec);
  free(reduced_vec);
  free(accum_vec);
  free(remote_vec);
  MPI_Finalize();
  return mismatch_counter > 0 ? 100 : 0;
}
