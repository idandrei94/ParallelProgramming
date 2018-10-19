/* In general MPI will handle all errors itself. It will print an
 * error message and abort all processes. Sometimes programmers
 * want to install their own error handlers or at least to handle
 * errors themselves to do some clean-up or to store an intermediate
 * state or whatever the reason is. If you overwrite the default
 * error handler "MPI_ERRORS_ARE_FATAL" with another error handler,
 * you must test all return values of MPI functions yourself. Usually
 * it is more convenient and more efficient to use the default error
 * handler.
 *
 * This program calls "MPI_Bcast ()" with a null communicator to
 * simulate an error. It evaluates the return value of the function.
 *
 *
 * Compiling:
 *   mpicc -o mpi_errors_2 mpi_errors_2.c
 *
 * Running:
 *   mpiexec -np 1 mpi_errors_2
 *
 *
 * File: mpi_errors_2.c			Author: S. Gross
 * Date: 19.07.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main (int argc, char *argv[])
{
  int  mytid,				/* my task id			*/
       value,				/* value to broadcast		*/
       err_string_length,		/* length of error string	*/
       ret;				/* return value of a function	*/
  char err_string[MPI_MAX_ERROR_STRING];

  MPI_Init (&argc, &argv);
  MPI_Comm_rank	(MPI_COMM_WORLD, &mytid);
  if (mytid == 0)
  {
    value = 10;				/* arbitrary value		*/
  }

  /* handle errors myself						*/
  printf ("Process %d: Set error handler for MPI_COMM_WORLD to "
	  "MPI_ERRORS_RETURN.\n", mytid);
  MPI_Comm_set_errhandler (MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  /* use wrong "root process" to produce an error			*/
  ret = MPI_Bcast (&value, 1, MPI_INT, -1, MPI_COMM_WORLD);
  if (ret != MPI_SUCCESS)
  {
    MPI_Error_string (ret, err_string, &err_string_length);
    fprintf (stderr, "Process %d: %s\n", mytid, err_string);
    MPI_Finalize ();
    exit (0);
  }
  /* reinstall default error handler				*/
  MPI_Comm_set_errhandler (MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
