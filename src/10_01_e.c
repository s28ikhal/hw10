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

long int compare_gather_remote(
    double * x_global, double * x_remote,     
    long int N, long int offset, long int n_per_rank,
    int remote_rank){
  long int mismatch_count = 0;
  for(long int i = offset; i < offset + n_per_rank; ++i){    
    if( fabs(x_global[i] - x_remote[i-offset]) > 2*DBL_EPSILON ){    
      printf("Mismatch: x_global[%ld] x_remote[%ld] (rank %d): %e %e\n",     
             i, i-offset, remote_rank, x_global[i], x_remote[i-offset]);
      mismatch_count++;
    }
  }
  return mismatch_count;
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

  long int vec_size = 32;
  if(argc > 1){
    vec_size = (long int) atol(argv[1]);
  } else {
    printf("Missing argument. Default vector size = %ld\n", vec_size);
  }
  if(vec_size % n_ranks != 0){
    if(rank == 0){
      printf("Vector size must be divisible by n_ranks = %d !\n", n_ranks);
    }
    mpi_exit(100);
  }

  long int local_vec_size = vec_size/n_ranks;

  double * remote_vec = NULL;
  double * local_vec = (double*) malloc(sizeof(double) * local_vec_size);
  double * global_vec = (double*) malloc(sizeof(double) * vec_size);

  // seed RNG in a rank-dependent manner
  // and fill local vector with random numbers
  srand(rank + 1234);
  for(long int i = 0; i < local_vec_size; ++i){
    local_vec[i] = ((double)rand())/RAND_MAX;
  }
  
  // gather vectors from all ranks in 'global_vec'
  MPI_Allgather(local_vec, local_vec_size, MPI_INT, global_vec, vec_size, MPI_INT, MPI_COMM_WORLD);

  // on rank 0, we are going to check that 'global_vec' contains the correct entries, need
  // a buffer for this purpose
  if( rank == 0 ){
    remote_vec = (double*) malloc(sizeof(double) * local_vec_size);
  }
 
  // iterate through all ranks and send 'local_vec' to rank 0, compare to
  // global_vec from Allgather above with correct offsets
  long int mismatch_count = 0;
  for(int r = 1; r < n_ranks; ++r){
    if(rank == 0){
      MPI_Recv(remote_vec, local_vec_size, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int offset = local_vec_size * r; // = ...; <- what's the correct offset?
      mismatch_count += compare_gather_remote(global_vec, remote_vec, vec_size, offset, local_vec_size, r); 
    } else if(rank == r) {
      MPI_Send(local_vec, local_vec_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }

  if( rank == 0 && mismatch_count == 0 ){
    printf("No mismatches.\n");
  }

  free(local_vec);
  free(global_vec);
  if(rank == 0 ){ 
    free(remote_vec); 
  };
  MPI_Finalize();
  return mismatch_count > 0 ? 101 : 0;
}
