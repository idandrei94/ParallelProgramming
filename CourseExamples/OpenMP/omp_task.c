/* This program computes a Fibonacci number with different methods.
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
 *   cc  -xopenmp -o omp_task omp_task.c
 *   gcc -fopenmp -o omp_task omp_task.c
 *   icc -qopenmp -o omp_task omp_task.c
 *   cl  /GL /Ox /openmp omp_task.c
 *
 * Running:
 *   ./omp_task
 *
 *
 * File: omp_task.c			Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define	NUMBER	   45			/* desired number		*/

/* function prototypes for the iterative and recursive version		*/
long long fibonacci_iterative (int n);
long long fibonacci_recursive (int n);
long long fibonacci_recursive_task (int n);


int main (void)
{
  long long fib_number	;		/* Fibonacci number		*/
  time_t    s_time, e_time;		/* start/end time of computation*/
  clock_t   cpu_time;			/* to measure computation time	*/


  printf ("\n\n"
	  "***************************************************\n"
	  "*** Iterative computation of Fibonacci numbers  ***\n"
	  "***************************************************\n");
  s_time   = time (NULL);
  cpu_time = clock ();
  fib_number = fibonacci_iterative (NUMBER);
  cpu_time = clock () - cpu_time;
  e_time   = time (NULL);
  printf ("fibonacci_iterative (%d) = %lld\n"
	  "\nTotal computation time: %6.2f\n"
	  "Total elapsed time:     %6.2f\n",
	  NUMBER, fib_number, (double) cpu_time / CLOCKS_PER_SEC,
	  difftime (e_time, s_time));

  printf ("\n\n"
	  "***************************************************\n"
	  "*** Recursive computation of Fibonacci numbers  ***\n"
	  "***************************************************\n");
  s_time   = time (NULL);
  cpu_time = clock ();
  fib_number = fibonacci_recursive (NUMBER);
  cpu_time = clock () - cpu_time;
  e_time   = time (NULL);
  printf ("fibonacci_recursive (%d) = %lld\n"
	  "\nTotal computation time: %6.2f\n"
	  "Total elapsed time:     %6.2f\n",
	  NUMBER, fib_number, (double) cpu_time / CLOCKS_PER_SEC,
	  difftime (e_time, s_time));

  printf ("\n\n"
	  "****************************************************"
	  "*****************\n"
	  "*** Recursive computation of Fibonacci numbers with "
	  "OpenMP tasks  ***\n"
	  "****************************************************"
	  "*****************\n");
  s_time   = time (NULL);
  cpu_time = clock ();
  #pragma omp parallel default(none) shared(fib_number)
  {
    /* It is important that "fibonacci_recursive_task (NUMBER)" will
     * only be called once so that all threads are available for
     * parallel "tasks". Otherwise every thread will call the function
     * and the Fibonacci number will be computed n times, if you use n
     * threads (resulting in more or less the same elapsed time, but
     * n times higher computation time than in the pure recursive
     * version without tasks).
     */
    #pragma omp single
    {
      printf ("Number of threads in parallel region: %d\n",
	      omp_get_num_threads ());
      fib_number = fibonacci_recursive_task (NUMBER);
    }
  }
  cpu_time = clock () - cpu_time;
  e_time   = time (NULL);
  printf ("fibonacci_recursive_task (%d) = %lld\n"
	  "\nTotal computation time: %6.2f\n"
	  "Total elapsed time:     %6.2f\n",
	  NUMBER, fib_number, (double) cpu_time / CLOCKS_PER_SEC,
	  difftime (e_time, s_time));

  return EXIT_SUCCESS;
}


/* Iterative computation of Fibonacci numbers. It doesn't make
 * sense to parallelize this version, because the loop doesn't
 * contain enough work.
 *
 * input parameters:	n	requested Fibonacci number
 * output parameters:	none
 * return value:	Fibonaci number
 * side effects:	terminates the program for "n < 0"
 *
 */
long long fibonacci_iterative (int n)
{
  if (n < 0)
  {
    fprintf (stderr, "Error: Parameter must be positive.\n"
	     "File: %s    Line: %d\n", __FILE__, __LINE__);
    exit (EXIT_FAILURE);
  }
  if ((n == 0) || (n == 1))
  {
    return (long long) n;
  }
  else
  {
    long long first, second, sum;	/* Fibonacci numbers		*/
    int	      i;			/* loop variable		*/

    first  = 0LL;
    second = 1LL;
    sum    = 0LL;
    for (i = 2; i <= n; ++i)
    {
      sum    = first + second;
      first  = second;
      second = sum;
    }
    return sum;
  }
}


/* Recursive computation of Fibonacci numbers.
 *
 * input parameters:	n	requested Fibonacci number
 * output parameters:	none
 * return value:	Fibonaci number
 * side effects:	terminates the program for "n < 0"
 *
 */
long long fibonacci_recursive (int n)
{
  if (n < 0)
  {
    fprintf (stderr, "Error: Parameter must be positive.\n"
	     "File: %s    Line: %d\n", __FILE__, __LINE__);
    exit (EXIT_FAILURE);
  }
  if ((n == 0) || (n == 1))
  {
    return (long long) n;
  }
  else
  {
    return fibonacci_recursive (n - 1) + fibonacci_recursive (n - 2);
  }
}


/* Recursive computation of Fibonacci numbers with OpenMP tasks.
 *
 * input parameters:	n	requested Fibonacci number
 * output parameters:	none
 * return value:	Fibonaci number
 * side effects:	terminates the program for "n < 0"
 *
 */
long long fibonacci_recursive_task (int n)
{
  long long fib1, fib2;

  if (n < 0)
  {
    fprintf (stderr, "Error: Parameter must be positive.\n"
	     "File: %s    Line: %d\n", __FILE__, __LINE__);
    exit (EXIT_FAILURE);
  }
  if ((n == 0) || (n == 1))
  {
    return (long long) n;
  }
  else
  {
    /* spawn a task							*/
    #pragma omp task shared(fib1, n) untied
    fib1 = fibonacci_recursive (n - 1);

    /* spawn another task						*/
    #pragma omp task shared(fib2, n) untied
    fib2 = fibonacci_recursive (n - 2);

    /* wait until both tasks have completed their work			*/
    #pragma omp taskwait
    return fib1 + fib2;
  }
}
