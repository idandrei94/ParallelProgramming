/* Small program that prints the sizes of MPI data types.
 *
 *
 * Compiling:
 *   mpicc -o mpi_datatype_sizes mpi_datatype_sizes.c
 *
 * Running:
 *   mpiexec -np 1 mpi_datatype_sizes
 *
 *
 * File: mpi_datatype_sizes.c		Author: S. Gross
 * Date: 19.07.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main (int argc, char *argv[])
{
  int  mytid,				/* my task id			*/
       size_mpi_char,			/* size of MPI data types	*/ 
       size_mpi_short,
       size_mpi_int,
       size_mpi_long,
       size_mpi_long_long_int,
       size_mpi_long_long,
       size_mpi_signed_char,
       size_mpi_unsigned_char,
       size_mpi_unsigned_short,
       size_mpi_unsigned,
       size_mpi_unsigned_long,
       size_mpi_unsigned_long_long,
       size_mpi_wchar,
       size_mpi_byte,
       size_mpi_float,
       size_mpi_double,
       size_mpi_long_double;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &mytid);
  MPI_Type_size (MPI_CHAR, &size_mpi_char);
  MPI_Type_size (MPI_SHORT, &size_mpi_short);
  MPI_Type_size (MPI_INT, &size_mpi_int);
  MPI_Type_size (MPI_LONG, &size_mpi_long);
  MPI_Type_size (MPI_LONG_LONG_INT, &size_mpi_long_long_int);
  MPI_Type_size (MPI_LONG_LONG, &size_mpi_long_long);
  MPI_Type_size (MPI_SIGNED_CHAR, &size_mpi_signed_char);
  MPI_Type_size (MPI_UNSIGNED_CHAR, &size_mpi_unsigned_char);
  MPI_Type_size (MPI_UNSIGNED_SHORT, &size_mpi_unsigned_short);
  MPI_Type_size (MPI_UNSIGNED, &size_mpi_unsigned);
  MPI_Type_size (MPI_UNSIGNED_LONG, &size_mpi_unsigned_long);
  MPI_Type_size (MPI_UNSIGNED_LONG_LONG, &size_mpi_unsigned_long_long);
  MPI_Type_size (MPI_WCHAR, &size_mpi_wchar);
  MPI_Type_size (MPI_BYTE, &size_mpi_byte);
  MPI_Type_size (MPI_FLOAT, &size_mpi_float);
  MPI_Type_size (MPI_DOUBLE, &size_mpi_double);
  MPI_Type_size (MPI_LONG_DOUBLE, &size_mpi_long_double);
  if (mytid == 0)
  {
    printf ("  sizeof (MPI_CHAR):               %d\n"
	    "  sizeof (MPI_SHORT):              %d\n"
	    "  sizeof (MPI_INT):                %d\n"
	    "  sizeof (MPI_LONG):               %d\n"
	    "  sizeof (MPI_LONG_LONG_INT):      %d\n"
	    "  sizeof (MPI_LONG_LONG):          %d\n"
	    "  sizeof (MPI_SIGNED_CHAR):        %d\n"
	    "  sizeof (MPI_UNSIGNED_CHAR):      %d\n"
	    "  sizeof (MPI_UNSIGNED_SHORT):     %d\n"
	    "  sizeof (MPI_UNSIGNED):           %d\n"
	    "  sizeof (MPI_UNSIGNED_LONG):      %d\n"
	    "  sizeof (MPI_UNSIGNED_LONG_LONG): %d\n"
	    "  sizeof (MPI_WCHAR):              %d\n"
	    "  sizeof (MPI_BYTE):               %d\n"
	    "  sizeof (MPI_FLOAT):              %d\n"
	    "  sizeof (MPI_DOUBLE):             %d\n"
	    "  sizeof (MPI_LONG_DOUBLE):        %d\n",
	    size_mpi_char, 
	    size_mpi_short, 
	    size_mpi_int, 
	    size_mpi_long, 
	    size_mpi_long_long_int, 
	    size_mpi_long_long, 
	    size_mpi_signed_char, 
	    size_mpi_unsigned_char, 
	    size_mpi_unsigned_short, 
	    size_mpi_unsigned, 
	    size_mpi_unsigned_long, 
	    size_mpi_unsigned_long_long, 
	    size_mpi_wchar, 
	    size_mpi_byte, 
	    size_mpi_float, 
	    size_mpi_double, 
	    size_mpi_long_double);
  }
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
