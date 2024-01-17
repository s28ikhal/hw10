#include <stdio.h>
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

  printf("This is rank id %d. The total number of MPI ranks is %d.\n", rank, n_ranks);

  MPI_Finalize();
  return 0;
}
