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
  mpi_error = MPI_Init(&argc, &argv);
  check_mpi_error(mpi_error, __FILE__, __LINE__-1);

  int rank = 0;
  int n_ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  // do not change the message
  const int message = rank * 11;
  int recv_buffer;

  // we are going to issue two MPI requests
  // -> need two each of MPI_Request and MPI_Status 
  MPI_Request requests[2];
  MPI_Status mpi_statuses[2];

  int rank_to = (rank+1) % n_ranks;
  int rank_from = (n_ranks + rank - 1) % n_ranks;

  MPI_Isend(&message, 1, MPI_INT, rank_to, 0, MPI_COMM_WORLD, &requests[0]);
  MPI_Irecv(&recv_buffer, 1, MPI_INT, rank_from, 0, MPI_COMM_WORLD, &requests[1]);

  MPI_Waitall(2, requests, mpi_statuses);

  // output everything in order
  for(int r = 0; r < n_ranks; ++r){
    if( rank == r ){
      printf("Rank %d sent %d to rank %d and received %d from rank %d\n"
             "mpi_statuses[0].MPI_SOURCE = %d, mpi_statuses[0].MPI_TAG = %d, mpi_statuses[0].MPI_ERROR = %d\n"
             "mpi_statuses[1].MPI_SOURCE = %d, mpi_statuses[1].MPI_TAG = %d, mpi_statuses[1].MPI_ERROR = %d\n\n",
             rank, message, rank_to, recv_buffer, mpi_statuses[1].MPI_SOURCE, 
             mpi_statuses[0].MPI_SOURCE, mpi_statuses[0].MPI_TAG, mpi_statuses[0].MPI_ERROR,
             mpi_statuses[1].MPI_SOURCE, mpi_statuses[1].MPI_TAG, mpi_statuses[1].MPI_ERROR);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}
