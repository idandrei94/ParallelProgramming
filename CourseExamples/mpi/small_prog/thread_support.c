/* The program shows the kind of thread support.
 *
 * If you initialize the MPI environment with "MPI_Init ()" it
 * supports only "MPI_SINGLE_THREAD". When you initialize it with
 * "MPI_Init_thread ()" it normally provides a better thread support
 * up to "MPI_THREAD_MULTIPLE".
 *
 *
 * Compiling:
 *   mpicc -o thread_support thread_support.c
 *
 * Running:
 *   mpiexec -np 1 thread_support
 *
 *
 * File: thread_support.c		       	Author: S. Gross
 * Date: 19.07.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main (int argc, char *argv[])
{
  int  thread_level,			/* kind of thread support	*/
       thread_is_main;			/* true if main thread		*/
  int  provided;			/* provided thread level	*/
  char *thr_level[] = {"MPI_THREAD_SINGLE",
		       "MPI_THREAD_FUNNELED",
		       "MPI_THREAD_SERIALIZED",
		       "MPI_THREAD_MULTIPLE"};

  MPI_Init_thread (&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if ((provided >= 0) &&
      (provided < (int) ((sizeof (thr_level) / sizeof (thr_level[0])))))
  {
    printf ("\n\nI have requested MPI_THREAD_MULTIPLE in "
	    "\"MPI_Init_thread ()\" and\n"
	    "it provides %s\n", thr_level[provided]);
  }
  else
  {
    printf ("\n\"MPI_Init_thread ()\" returned unknow thread level "
	    "%d.\n\n", provided);
  }

  MPI_Query_thread (&thread_level);
  printf ("\n\"MPI_Query_thread ()\" returned ");
  switch (thread_level)
  {
    case MPI_THREAD_SINGLE:
      printf ("MPI_THREAD_SINGLE:\n"
	      "\tOnly one thread supported.\n");
      break;

    case MPI_THREAD_FUNNELED:
      printf ("MPI_THREAD_FUNNELED:\n"
	      "\tMany threads are allowed, but only the main thread\n"
	      "\tmay call MPI functions.\n");
      break;

    case MPI_THREAD_SERIALIZED:
      printf ("MPI_THREAD_SERIALIZED:\n"
	      "\tMany threads may call MPI functions, but only one\n"
	      "\tthread at a time (the user must guarantee this).\n");
      break;

    case MPI_THREAD_MULTIPLE:
      printf ("MPI_THREAD_MULTIPLE:\n"
	      "\tMany threads are supported and any thread may call\n"
	      "\tMPI functions at any time.\n");
      break;

    default:
      printf ("\nUnknown thread support\n");
  }

  MPI_Is_thread_main (&thread_is_main);
  printf ("\n\"MPI_Is_thread_main ()\" returned: ");
  if (thread_is_main != 0)
  {
    printf ("\"true\".\n\n");
  }
  else
  {
    printf ("\"false\".\n\n");
  }
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
