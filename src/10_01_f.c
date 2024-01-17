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
    int remote_rank, int local_rank){
  long int mismatch_count = 0;
  for(long int i = offset; i < offset + n_per_rank; ++i){    
    if( fabs(x_global[i] - x_remote[i-offset]) > 2*DBL_EPSILON ){    
      printf("Mismatch: x_global[%ld] x_remote[%ld] (from rank %d on rank %d): %e %e\n",    
             i, i-offset, remote_rank, local_rank, x_global[i], x_remote[i-offset]);    
      mismatch_count++;    
    }    
  }    
  return mismatch_count;    
}    

long int count_mismatches(double * global_vec, double *local_vec,
                     long int N, long int n_per_rank, int rank, int n_ranks){
  long int mismatch_counter = 0;
  double * remote_vec = malloc(n_per_rank * sizeof(double));

  for(int r1 = 0; r1 < n_ranks; ++r1){
    for(int r2 = 0; r2 < n_ranks; ++r2){
      if( r1 == r2 ) {
        continue;
      } else {
        if( r1 == rank ){
          MPI_Recv(remote_vec, n_per_rank, MPI_DOUBLE, r2, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          mismatch_counter += compare_gather_remote(global_vec, remote_vec, N, r2*n_per_rank, n_per_rank, r2, rank); 
        } else if ( r2 == rank) {
          MPI_Send(local_vec, n_per_rank, MPI_DOUBLE, r1, 42, MPI_COMM_WORLD);
        }
      }
    }
    // this Barrier is only here to order the output of 'compare_gather_remote'
    // it is not technically necessary
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  free(remote_vec);
  return mismatch_counter;
}

int main(int argc, char ** argv)
{
  // in principle almost every MPI function returns an integer to communicate
  // the success or failure of the operation
  int mpi_error;
  //mpi_error = MPI_Init(...); 
  check_mpi_error(mpi_error, __FILE__, __LINE__-1);

  int rank = 0;
  int n_ranks = 1;
  //MPI_Comm_rank(...);
  //MPI_Comm_size(...);

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

  double * local_vec = (double*) malloc(sizeof(double) * local_vec_size);
  double * global_vec = (double*) malloc(sizeof(double) * vec_size);

  srand(rank + 1234);
  // fill local_vec with rank-dependent pseudo-random-numbers
  for(long int i = 0; i < local_vec_size; ++i){
    local_vec[i] = ((double)rand())/RAND_MAX;
  }

  // we are going to issue 2*(n_ranks-1) MPI requests, need storage for these
  MPI_Status * statuses = n_ranks > 1 ? (MPI_Status*) malloc(2*(n_ranks-1)*sizeof(MPI_Status)) : NULL;
  MPI_Request * requests = n_ranks > 1 ? (MPI_Request*) malloc(2*(n_ranks-1)*sizeof(MPI_Request)) : NULL;

  int request_counter = 0;
  for(int r = 0; r < n_ranks; ++r){
    // use memcpy instead of communication for the current rank
    if( r == rank ){
      memcpy(global_vec + rank*local_vec_size, local_vec, local_vec_size*sizeof(double));
    } else {
      //MPI_Isend(local_vec, ..., &requests[request_counter]);
      request_counter++;
      long int offset;  // <- what is the correct offset?
      //MPI_Irecv(global_vec + offset, ..., &requests[request_counter]);
      request_counter++;
    }
  }
  if(n_ranks > 1 ) MPI_Waitall(request_counter, requests, statuses);

  long int mismatch_counter = count_mismatches(global_vec, local_vec, vec_size, local_vec_size, rank, n_ranks);

  if( mismatch_counter == 0 ){
    printf("Rank %d: no mismatches\n", rank);
  }

  free(local_vec);
  free(global_vec);
  free(statuses);
  free(requests);
  MPI_Finalize();
  return mismatch_counter > 0 ? 101 : 0;
}
