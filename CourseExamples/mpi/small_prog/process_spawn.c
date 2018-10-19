/* The program can be used to measure the time to create some
 * processes via the command line with "mpiexec" or dynamically
 * with "MPI_Comm_spawn ()". It uses the argument vector to
 * determine the name of the process (argv[0]) and the number of
 * processes it shall create (argv[1]), if "mpiexec" started only
 * one process.
 *
 *
 * Compiling:
 *   mpicc -o process_spawn process_spawn.c
 *
 * Running:
 *   mpiexec -np <number of processes> process_spawn
 *
 *
 * File: process_spawn.c		Author: S. Gross
 * Date: 19.07.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main (int argc, char *argv[])
{
  MPI_Comm COMM_CHILD_PROCESSES,	/* inter-communicator		*/
	   COMM_PARENT_PROCESSES;	/* inter-communicator		*/
  int	   ntasks_world,		/* # of tasks in MPI_COMM_WORLD	*/
	   mytid_world,			/* my task id in MPI_COMM_WORLD	*/
	   num_procs;			/* # of processes to create	*/

  MPI_Init (&argc, &argv);
  MPI_Comm_rank	(MPI_COMM_WORLD, &mytid_world);
  MPI_Comm_size	(MPI_COMM_WORLD, &ntasks_world);
  /* At first we must decide if this program is executed from a parent
   * or child process because only a parent is allowed to spawn child
   * processes (otherwise the child process with rank 0 would spawn
   * itself child processes and so on). "MPI_Comm_get_parent ()"
   * returns the parent inter-communicator for a spawned MPI rank and
   * MPI_COMM_NULL if the process wasn't spawned, i.e. it was started
   * statically via "mpiexec" on the command line.
   */
  MPI_Comm_get_parent (&COMM_PARENT_PROCESSES);
  if (COMM_PARENT_PROCESSES == MPI_COMM_NULL)
  {
    /* parent processes have work to do in this program			*/
    if (ntasks_world == 1)
    {
      /* create processes with "MPI_Comm_spawn ()"			*/
      if (argc == 2)
      {
	num_procs = atoi (argv[1]);
      }
      else
      {
	fprintf (stderr, "\n\nError: Wrong number of parameters.\n\n"
		 "Usage:\n"
		 "  time mpiexec -np <number of processes> %s\n"
		 "or\n"
		 "  time mpiexec -np 1 %s <number of processes>\n"
		 "or\n"
		 "  /usr/bin/time mpiexec -np 1 %s "
		 "<number of processes>\n",
		 argv[0], argv[0], argv[0]);
	MPI_Finalize ();
	exit (EXIT_SUCCESS);
      }
      /* All parent processes must call "MPI_Comm_spawn ()" but only
       * the root process (in our case the process with rank 0) will
       * spawn child processes. All other processes of the
       * intra-communicator (in our case MPI_COMM_WORLD) will ignore
       * the values of all arguments before the "root" parameter.
       */
      MPI_Comm_spawn (argv[0], MPI_ARGV_NULL, num_procs,
		      MPI_INFO_NULL, 0, MPI_COMM_WORLD,
		      &COMM_CHILD_PROCESSES, MPI_ERRCODES_IGNORE);
      printf ("\nOne process created via \"mpiexec\" and %d processes "
	      "with \"MPI_Comm_spawn ()\".\n\n", num_procs);
    }
    else
    {
      /* all processes have been created via the command line with
       * "mpiexec"
       */
      if (mytid_world == 0)
      {
	printf ("\n%d processes created via \"mpiexec\".\n\n",
		ntasks_world);
      }
    }
  }
  else
  {
    /* child processes have nothing to do in this program		*/
  }
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
