/* The program demonstrates how to initialize and finalize an
 * MPI environment.
 *
 *
 * Compiling:
 *   mpicc -o init_finalize init_finalize.c
 *
 * Running:
 *   mpiexec -np <number of processes> init_finalize
 *
 *
 * File: init_finalize.c		       	Author: S. Gross
 * Date: 19.07.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main (int argc, char *argv[])
{
  MPI_Init (&argc, &argv);
  /* With the next statement every process executing this code will
   * print one line on the display. It may happen that the lines will
   * get mixed up because the display is a critical section. In general
   * only one process (mostly the process with rank 0) will print on
   * the display and all other processes will send their messages to
   * this process. Nevertheless for debugging purposes (or to
   * demonstrate that it is possible) it may be useful if every
   * process prints itself.
   */
  printf ("Hello!\n");
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
