#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

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

  // do not modify message
  const int message = rank * 11;
  int recv_buffer;
  MPI_Status mpi_status;

  int rank_to;
  int rank_from;

  //MPI_Sendrecv(..., &mpi_status);

  printf("Rank %d sent %d to rank %d and received %d from rank %d, tagged %d with error value %d\n",
         rank, message, rank_to, recv_buffer, mpi_status.MPI_SOURCE, mpi_status.MPI_TAG, mpi_status.MPI_ERROR);

  MPI_Finalize();
  return 0;
}
