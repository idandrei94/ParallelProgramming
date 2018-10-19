/* Parallel initialization of a triangular matrix  can result in
 * heavy load imbalances. Use the "schedule clause" to improve
 * load balancing.
 *
 * This version uses OpenMP.
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
 *   cc  -xopenmp -o triangular_matrix triangular_matrix.c [-lm]
 *   gcc -fopenmp -o triangular_matrix triangular_matrix.c [-lm]
 *   icc -qopenmp -o triangular_matrix triangular_matrix.c
 *   cl  /GL /Ox /openmp triangular_matrix.c
 *
 * Running:
 *   ./triangular_matrix
 *
 *
 * File: triangular_matrix.c		Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <omp.h>

#define	P 1000				/* # of rows / columns		*/

/* a matrix of this size may be too large for a normal stack size and
 * must be allocated globally or with the keyword "static".
 */
static double a[P][P];			/* for upper triangular matrix	*/

int main (void)
{
  int *cnt_elem,			/* # of matrix elements/thread	*/
      num_threads,			/* # of threads			*/
      my_thr_id,			/* own thread number		*/
      i, j;				/* loop variables		*/

  #pragma omp parallel default(none) private(i, j, my_thr_id) \
    shared(a, cnt_elem, num_threads)
  {
    /* get number of threads and allocate memory for the vector
     * which counts the number of elements each thread processes
     */
    #pragma omp single
    {
      num_threads = omp_get_num_threads ();
      cnt_elem = (int *) malloc ((size_t) num_threads * sizeof (int));
      if (cnt_elem == NULL)
      {
	/* If you use
	 *
	 * fprintf (stderr, "File: %s, line %d: Can't allocate memory.\n",
	 *	    __FILE__, __LINE__);
	 *
	 * you get an error message similar to
	 *
	 * triangular_matrix.c:72:11: error: '__iob' not specified in
	 *   enclosing parallel
	 *
	 * Use "perror ()" to avoid compiler specific variables in the
	 * enclosing parallel clause.
	 */
	perror ("Can't allocate memory.");
	exit (EXIT_FAILURE);
      }
      for (i = 0; i < num_threads; ++i)
      {
	cnt_elem[i] = 0;
      }
    }
    
    my_thr_id = omp_get_thread_num ();
    /* initialize upper triangular matrix				*/
    #pragma omp for schedule(runtime)
    for (i = 0; i < P; ++i)
    {
      for (j = i; j < P; ++j)
      {
	a[i][j] = sqrt ((i + 1.0) * (j + 1.0));
	cnt_elem[my_thr_id]++;
      }
    }
  }
  /* use somthing from matrix "a" so that the compiler will not
   * remove the above loops as dead code.
   */
  printf ("a[%d][%d] = %g\n"
	  "a[%d][%d] = %g\n",
	  0, 0, a[0][0], P - 1, P - 1, a[P -1][P -1]);
  /* print results							*/
  printf ("Number of elements in upper triangular matrix: %d\n"
	  "Number of threads: %d\n",
	  (P * (P + 1)) / 2, num_threads);
  for (i = 0; i < num_threads; ++i)
  {
    printf ("Thread %d initialized %d matrix elements.\n",
	    i, cnt_elem[i]);
  }
  free (cnt_elem);
  return EXIT_SUCCESS;
}
