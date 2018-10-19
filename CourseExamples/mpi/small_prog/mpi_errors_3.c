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
 * simulate an error. It handles all errors itself via an error
 * handler.
 *
 *
 * Compiling:
 *   mpicc -o mpi_errors_3 mpi_errors_3.c
 *
 * Running:
 *   mpiexec -np 1 mpi_errors_3
 *
 *
 * File: mpi_errors_3.c			Author: S. Gross
 * Date: 19.07.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

/* error handler for communicators					*/
void my_comm_error_handler (MPI_Comm *comm, int *err_code, ...);


int main (int argc, char *argv[])
{
  int  mytid,				/* my task id			*/
       value;				/* value to broadcast		*/
  MPI_Errhandler comm_errhandler;	/* communicator error handler	*/

  MPI_Init (&argc, &argv);
  MPI_Comm_rank	(MPI_COMM_WORLD, &mytid);
  if (mytid == 0)
  {
    value = 10;				/* arbitrary value		*/
  }

  /* handle errors myself						*/
  printf ("Process %d: Set error handler for MPI_COMM_WORLD to "
	  "my own error handler.\n", mytid);
  MPI_Comm_create_errhandler (my_comm_error_handler, &comm_errhandler);
  MPI_Comm_set_errhandler (MPI_COMM_WORLD, comm_errhandler);
  /* use wrong "root process" to produce an error			*/
  MPI_Bcast (&value, 1, MPI_INT, -1, MPI_COMM_WORLD);
  /* reinstall default error handler and release own error handler	*/
  MPI_Comm_set_errhandler (MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
  MPI_Errhandler_free (&comm_errhandler);
  MPI_Finalize ();
  return EXIT_SUCCESS;
}


/* The current MPI error handler is called to handle errors, if they
 * occur in an MPI function call. You can attach this error handler
 * to a communicator. The first parameter is the communicator in use
 * and the second one contains the error code which was returned from
 * the MPI function that raised the error.
 *
 * input parameters:	comm		address of communicator
 *			err_code	address of error code
 * output parameters:	none
 * return value:	none
 * side effects:	terminates the process
 *
 */
void my_comm_error_handler (MPI_Comm *comm, int *err_code, ...)
{
  int  cmp_result,			/* result of comparison		*/
       err_string_length;		/* length of error string	*/
  char err_string[MPI_MAX_ERROR_STRING];

  /* this demonstrates how you can use parameter "comm" to distinguish
   * between different communicators
   */
  MPI_Comm_compare (*comm, MPI_COMM_WORLD, &cmp_result);
  if (cmp_result == MPI_IDENT)
  {
    MPI_Error_string (*err_code, err_string, &err_string_length);
    fprintf (stderr, "\nError for communicator MPI_COMM_WORLD:\n"
	     "  %s\n\n", err_string);
    MPI_Finalize ();
    exit (0);
  }

  /* You will reach this point only for unexpected communicators	*/
  fprintf (stderr, "\nError in \"my_comm_error_handler ()\": "
	   "unexpected communicator.\n");
  MPI_Finalize ();
  exit (0);
}
