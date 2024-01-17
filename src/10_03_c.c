#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>
#include <string.h>

/* For LONG_MAX */
#include <limits.h>

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

int main(int argc, char* argv[])
{
  int mpi_error;
  int ranks_per_color = 1;

  if(argc > 1)
    ranks_per_color = atoi(argv[1]);
  else
    printf("Missing 1st argument. Default ranks_per_color = %d\n", ranks_per_color);
  
  mpi_error = MPI_Init(&argc, &argv);
  check_mpi_error(mpi_error, __FILE__, __LINE__-1);

  int world_rank = 0;
  int n_ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  if( n_ranks % ranks_per_color != 0 ){
    if( world_rank == 0 ) printf("Total number of ranks %d must be divisible by ranks_per_color %d! Exiting.\n",
                                 n_ranks, ranks_per_color);
    mpi_exit(101);
  }

  int color; // <- color assigment
  int color_rank; // <- rank assignment

  // split the MPI_COMM_WORLD communicator into appropriate color_comm communicators
  MPI_Comm color_comm;
  //MPI_Comm_split(MPI_COMM_WORLD, ...);

  int color_comm_size;
  int color_comm_rank;
  //MPI_Comm_rank(color_comm, ...);
  //MPI_Comm_size(color_comm, ...);

  printf("Rank %d (world) has rank %d in its color_comm (color %d), which has size %d\n",
         world_rank, color_comm_rank, color, color_comm_size);

  MPI_Finalize();
  return 0;
}
