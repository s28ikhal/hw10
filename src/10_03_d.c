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
  int n_ranks_per_row = 1;

  if(argc > 1)
    n_ranks_per_row = atoi(argv[1]);
  else
    printf("Missing 1st argument. Default n_ranks_per_row = %d\n", n_ranks_per_row);
  
  mpi_error = MPI_Init(&argc, &argv);
  check_mpi_error(mpi_error, __FILE__, __LINE__-1);

  int world_rank = 0;
  int n_ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  if( n_ranks % n_ranks_per_row != 0 ){
    if( world_rank == 0 ) printf("Total number of ranks %d must be divisible by n_ranks_per_row %d! Exiting.\n",
                                 n_ranks, n_ranks_per_row);
    mpi_exit(100);
  }

  int row_color; // = row color
  int row_color_rank; // = rowwise rank assignment

  int col_color; // = column color
  int col_color_rank; // = columnwise rank assignment

  // rowwise and columnwise split communicators
  MPI_Comm row_comm, col_comm;

  // split MPI_COMM_WORLD rowwise as in exercise 10_03_c into row_comm
  //MPI_Comm_split(MPI_COMM_WORLD, ...);
  
  // split MPI_COMM_WORLD columnwise into col_comm
  //MPI_Comm_split(MPI_COMM_WORLD, ...);
  
  // extract rank id and size in row_comm
  int row_comm_rank, row_comm_size;
  //MPI_Comm_rank(row_comm, ...);
  //MPI_Comm_size(row_comm, ...);
  
  // extract rank id and size in col_comm
  int col_comm_rank, col_comm_size;
  //MPI_Comm_rank(col_comm, ...);
  //MPI_Comm_size(col_comm, ...);

  if( world_rank == 0 ){
    printf("( world_rank [row_color,row_comm_rank] [col_color,col_comm_rank] )\n\n");
    fflush(stdout);
  }

  // ordered output of rank assignments
  MPI_Barrier(MPI_COMM_WORLD);
  for(int r = 0; r < n_ranks; r++){
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    if( world_rank == r ){
      printf("(%3d [%2d,%2d] [%2d,%2d])  ",
             world_rank, row_color, row_comm_rank, col_color, col_comm_rank);
      fflush(stdout);
      if(r > 0 && (r+1) % n_ranks_per_row == 0 ) printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(world_rank == 0 ) printf("\n\n");

  MPI_Finalize();
  return 0;
}
