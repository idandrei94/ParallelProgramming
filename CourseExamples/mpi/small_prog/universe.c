/* The program shows how many virtual cpu's are available in the MPI
 * environment. This is optional and an implementation doesn't need
 * to provide this information.
 *
 *
 * Compiling:
 *   mpicc -o universe universe.c
 *
 * Running:
 *   mpiexec -np <number of processes> universe
 *
 *
 * File: universe.c		       	Author: S. Gross
 * Date: 19.07.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main (int argc, char *argv[])
{
  int  ntasks,				/* number of parallel tasks	*/
       mytid,				/* my task id			*/
       *universe_size_ptr,		/* ptr to # of "virtual cpu's"	*/
       default_universe_size = 1,
       universe_size_flag;		/* true if available		*/

  universe_size_ptr = &default_universe_size;
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &mytid);
  MPI_Comm_size (MPI_COMM_WORLD, &ntasks);
  MPI_Comm_get_attr (MPI_COMM_WORLD, MPI_UNIVERSE_SIZE,
		     &universe_size_ptr, &universe_size_flag);
  if (mytid == 0)
  {
    if (universe_size_flag != 0)
    {
      printf ("\nnumber of processes: %d   universe size: %d\n\n",
	      ntasks, *universe_size_ptr);
    }
    else
    {
      printf ("\n\"MPI_UNIVERSE_SIZE\" not available. "
	      "I use my default value.\n"
	      "number of processes: %d   universe size: %d\n\n",
	      ntasks, *universe_size_ptr);
    }
  }
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
