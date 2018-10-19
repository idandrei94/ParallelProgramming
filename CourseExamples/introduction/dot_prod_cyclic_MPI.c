/* Compute the dot product of two vectors in parallel with MPI.
 * Every process is cycling through the index space.
 *
 * Compiling:
 *   mpicc -o dot_prod_cyclic_MPI dot_prod_cyclic_MPI.c
 *
 * Running:
 *   mpiexec -np <number of processes> dot_prod_cyclic_MPI
 *
 *
 * File: dot_prod_cyclic_MPI.c		Author: S. Gross
 * Date: 03.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define VECTOR_SIZE 100000000		/* vector size (10^8)		*/

/* heap memory to avoid a segmentation fault due to a stack overflow	*/
static double a[VECTOR_SIZE],		/* vectors for dot product	*/
	      b[VECTOR_SIZE];


int main (int argc, char *argv[])
{
  int mytid,				/* task id (process rank)	*/
      ntasks;				/* number of processes		*/
  double sum,
	 my_sum;			/* partial sum of one process	*/

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &mytid);
  MPI_Comm_size (MPI_COMM_WORLD, &ntasks);

  /* initialize vectors							*/
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
    a[i] = 2.0;
    b[i] = 3.0;
  }

  /* compute dot product						*/
  my_sum = 0.0;
  for (int i = mytid; i < VECTOR_SIZE; i += ntasks)
  {
    #if VECTOR_SIZE < 20
      printf ("Process %d: i = %d.\n", mytid, i);
    #endif
    my_sum += a[i] * b[i];
  }
  MPI_Reduce (&my_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (mytid == 0)
  {
    printf ("sum = %e\n", sum);
  }
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
