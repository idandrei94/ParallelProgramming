/* Small program that creates a vector and prints its size and
 * extent.
 *
 * An MPI data type is defined by its size, its contents, and its
 * extent. When multiple elements of the same size are used in a
 * contiguous manner (e.g. in a "scatter" operation or an operation
 * with "count" greater than one) the extent is used to compute where
 * the next element will start. The extent for a derived data type is
 * as big as the size of the derived data type so that the first
 * elements of the second structure will start after the last element
 * of the first structure, i.e., you have to "resize" the new data
 * type if you want to send it multiple times (count > 1) or to
 * scatter/gather it to many processes. Restrict the extent of the
 * derived data type for a strided vector in such a way that it looks
 * like just one element if it is used with "count > 1" or in a
 * scatter/gather operation.
 *
 *
 * Compiling:
 *   mpicc -o size_extent size_extent.c
 *
 * Running:
 *   mpiexec -np 1 size_extent
 *
 *
 * File: size_extent.c			Author: S. Gross
 * Date: 19.07.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define COUNT		2
#define	BLOCKLENGTH	2
#define STRIDE		4

int main (int argc, char *argv[])
{
  int  mytid,				/* my task id			*/
       size,				/* size of vector		*/
       size_mpi_double;			/* size of MPI_DOUBLE		*/
  MPI_Aint lb, extent,			/* lower bound, extent of vector*/
	   true_lb, true_extent;
  MPI_Datatype vector_t,		/* strided vector		*/
	       tmp_vector_t;		/* needed to resize the extent	*/

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &mytid);
  MPI_Type_size (MPI_DOUBLE, &size_mpi_double);

  /* Build the new type for a strided vector and resize the extent of
   * the new datatype in such a way that the extent of the whole vector
   * looks like just one element.
   */
  MPI_Type_vector (COUNT, BLOCKLENGTH, STRIDE, MPI_DOUBLE,
		   &tmp_vector_t);
  MPI_Type_create_resized (tmp_vector_t, 0, size_mpi_double, &vector_t);
  MPI_Type_commit (&vector_t);
  MPI_Type_free (&tmp_vector_t);
  if (mytid == 0)
  {
    MPI_Type_size (vector_t, &size);
    MPI_Type_get_extent (vector_t, &lb, &extent);
    MPI_Type_get_true_extent (vector_t, &true_lb, &true_extent);
    printf ("strided vector:\n"
	    "  size of old data type: %d\n"
	    "  count:                 %d\n"
	    "  blocklength:           %d\n"
	    "  stride:                %d\n"
	    "  size:                  %d\n"
	    "  lower bound:           %ld\n"
	    "  extent:                %ld\n"
	    "  true lb:               %ld\n"
	    "  true extent:           %ld\n",
	    size_mpi_double, COUNT, BLOCKLENGTH,
	    STRIDE, size, (long) lb, (long) extent, (long) true_lb,
	    (long) true_extent);
  }
  MPI_Type_free (&vector_t);
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
