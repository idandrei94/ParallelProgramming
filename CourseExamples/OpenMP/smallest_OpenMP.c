/* The smallest OpenMP program to display "Hello World" on the screen.
 *
 * The default number of parallel threads depends on the
 * implementation, e.g., just one thread or one thread for every
 * virtual processor. You can for example request four threads, if
 * you set the environment variable "OMP_NUM_THREADS" to "4" before
 * you run the program, e.g., "setenv OMP_NUM_THREADS 4". If you
 * compile the program with the Oracle C compiler (former Sun C
 * compiler) the number of threads is reduced to the number of
 * virtual processors by default if the number of requested threads
 * is greater than the number of virtual processors. You can change
 * this behaviour if you set the environment variable "OMP_DYNAMIC"
 * to "FALSE" before you run the program, e.g., "setenv OMP_DYNAMIC
 * FALSE".
 *
 *
 * Compiling:
 *   cc  -xopenmp -o smallest_OpenMP smallest_OpenMP.c
 *   gcc -fopenmp -o smallest_OpenMP smallest_OpenMP.c
 *   icc -qopenmp -o smallest_OpenMP smallest_OpenMP.c
 *   cl  /GL /Ox /openmp smallest_OpenMP.c
 *
 * Running:
 *   ./smallest_OpenMP
 *
 *
 * File: smallest_OpenMP.c		Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>

int main (void)
{
  #pragma omp parallel
  {
    printf ("Hello!\n");
  }
  return EXIT_SUCCESS;
}
