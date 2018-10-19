/* The program demonstrates how a process can determine the number
 * of programs within the MPI environment, the name of the processor
 * which it uses, and how it can find out its own rank within the
 * process group. Furthermore it prints the version of the implemented
 * MPI standard.
 *
 *
 * Compiling:
 *   mpicc -o rank_size rank_size.c
 *
 * Running:
 *   mpiexec -np <number of processes> rank_size
 *
 *
 * File: rank_size.c		       	Author: S. Gross
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
       version, subversion,		/* version of MPI standard	*/
       namelen;				/* length of processor name	*/
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &mytid);
  MPI_Comm_size (MPI_COMM_WORLD, &ntasks);
  MPI_Get_processor_name (processor_name, &namelen);
  /* With the next statement every process executing this code will
   * print one line on the display. It may happen that the lines will
   * get mixed up because the display is a critical section. In general
   * only one process (mostly the process with rank 0) will print on
   * the display and all other processes will send their messages to
   * this process. Nevertheless for debugging purposes (or to
   * demonstrate that it is possible) it may be useful if every
   * process prints itself.
   */
  printf ("I'm process %d of %d available processes running on %s.\n",
	  mytid, ntasks, processor_name);
  MPI_Get_version (&version, &subversion);
  printf ("MPI standard %d.%d is supported.\n", version, subversion);
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
