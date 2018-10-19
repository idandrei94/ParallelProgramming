/* Small program that prints the extents of MPI data types.
 *
 *
 * Compiling:
 *   mpicc -o mpi_datatype_extents mpi_datatype_extents.c
 *
 * Running:
 *   mpiexec -np 1 mpi_datatype_extents
 *
 *
 * File: mpi_datatype_extents.c		Author: S. Gross
 * Date: 19.07.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main (int argc, char *argv[])
{
  int	   mytid;			/* my task id			*/
  MPI_Aint extent_mpi_char,		/* extent of MPI data types	*/ 
	   extent_mpi_short,
	   extent_mpi_int,
	   extent_mpi_long,
	   extent_mpi_long_long_int,
	   extent_mpi_long_long,
	   extent_mpi_signed_char,
	   extent_mpi_unsigned_char,
	   extent_mpi_unsigned_short,
	   extent_mpi_unsigned,
	   extent_mpi_unsigned_long,
	   extent_mpi_unsigned_long_long,
	   extent_mpi_wchar,
	   extent_mpi_byte,
	   extent_mpi_float,
	   extent_mpi_double,
	   extent_mpi_long_double,
	   lb_mpi_char,			/* lower bound of MPI data types*/ 
	   lb_mpi_short,
	   lb_mpi_int,
	   lb_mpi_long,
	   lb_mpi_long_long_int,
	   lb_mpi_long_long,
	   lb_mpi_signed_char,
	   lb_mpi_unsigned_char,
	   lb_mpi_unsigned_short,
	   lb_mpi_unsigned,
	   lb_mpi_unsigned_long,
	   lb_mpi_unsigned_long_long,
	   lb_mpi_wchar,
	   lb_mpi_byte,
	   lb_mpi_float,
	   lb_mpi_double,
	   lb_mpi_long_double;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &mytid);
  MPI_Type_get_extent (MPI_CHAR, &lb_mpi_char, &extent_mpi_char);
  MPI_Type_get_extent (MPI_SHORT, &lb_mpi_short, &extent_mpi_short);
  MPI_Type_get_extent (MPI_INT, &lb_mpi_int, &extent_mpi_int);
  MPI_Type_get_extent (MPI_LONG, &lb_mpi_long, &extent_mpi_long);
  MPI_Type_get_extent (MPI_LONG_LONG_INT, &lb_mpi_long_long_int,
		       &extent_mpi_long_long_int);
  MPI_Type_get_extent (MPI_LONG_LONG, &lb_mpi_long_long,
		       &extent_mpi_long_long);
  MPI_Type_get_extent (MPI_SIGNED_CHAR, &lb_mpi_signed_char,
		       &extent_mpi_signed_char);
  MPI_Type_get_extent (MPI_UNSIGNED_CHAR, &lb_mpi_unsigned_char,
		       &extent_mpi_unsigned_char);
  MPI_Type_get_extent (MPI_UNSIGNED_SHORT, &lb_mpi_unsigned_short,
		       &extent_mpi_unsigned_short);
  MPI_Type_get_extent (MPI_UNSIGNED, &lb_mpi_unsigned,
		       &extent_mpi_unsigned);
  MPI_Type_get_extent (MPI_UNSIGNED_LONG, &lb_mpi_unsigned_long,
		       &extent_mpi_unsigned_long);
  MPI_Type_get_extent (MPI_UNSIGNED_LONG_LONG,
		       &lb_mpi_unsigned_long_long,
		       &extent_mpi_unsigned_long_long);
  MPI_Type_get_extent (MPI_WCHAR, &lb_mpi_wchar, &extent_mpi_wchar);
  MPI_Type_get_extent (MPI_BYTE, &lb_mpi_byte, &extent_mpi_byte);
  MPI_Type_get_extent (MPI_FLOAT, &lb_mpi_float, &extent_mpi_float);
  MPI_Type_get_extent (MPI_DOUBLE, &lb_mpi_double, &extent_mpi_double);
  MPI_Type_get_extent (MPI_LONG_DOUBLE, &lb_mpi_long_double,
		       &extent_mpi_long_double);
  if (mytid == 0)
  {
    printf ("  extent of MPI_CHAR:               %d\n"
	    "  extent of MPI_SHORT:              %d\n"
	    "  extent of MPI_INT:                %d\n"
	    "  extent of MPI_LONG:               %d\n"
	    "  extent of MPI_LONG_LONG_INT:      %d\n"
	    "  extent of MPI_LONG_LONG:          %d\n"
	    "  extent of MPI_SIGNED_CHAR:        %d\n"
	    "  extent of MPI_UNSIGNED_CHAR:      %d\n"
	    "  extent of MPI_UNSIGNED_SHORT:     %d\n"
	    "  extent of MPI_UNSIGNED:           %d\n"
	    "  extent of MPI_UNSIGNED_LONG:      %d\n"
	    "  extent of MPI_UNSIGNED_LONG_LONG: %d\n"
	    "  extent of MPI_WCHAR:              %d\n"
	    "  extent of MPI_BYTE:               %d\n"
	    "  extent of MPI_FLOAT:              %d\n"
	    "  extent of MPI_DOUBLE:             %d\n"
	    "  extent of MPI_LONG_DOUBLE:        %d\n",
	    (int) extent_mpi_char, 
	    (int) extent_mpi_short, 
	    (int) extent_mpi_int, 
	    (int) extent_mpi_long, 
	    (int) extent_mpi_long_long_int, 
	    (int) extent_mpi_long_long, 
	    (int) extent_mpi_signed_char, 
	    (int) extent_mpi_unsigned_char, 
	    (int) extent_mpi_unsigned_short, 
	    (int) extent_mpi_unsigned, 
	    (int) extent_mpi_unsigned_long, 
	    (int) extent_mpi_unsigned_long_long, 
	    (int) extent_mpi_wchar, 
	    (int) extent_mpi_byte, 
	    (int) extent_mpi_float, 
	    (int) extent_mpi_double, 
	    (int) extent_mpi_long_double);
  }
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
