/* Compute the dot product of two vectors in parallel with Pthreads.
 * Every thread works on a block  of the index space.
 *
 * If you want to compile a program in 64-bit mode you have to solve
 * a small problem with "pthread_create" and "pthread_join" which
 * expect a function with a parameter "void *" returning "void *".
 * In 32-bit programs (ILP32) the types "int", "long", and "pointer"
 * are all 32 bits so that it isn't a problem to store a pointer in an
 * integer variable or convert a pointer to an integer if you want a
 * function with parameter "int" and/or returning "int". Unfortunately
 * this is no longer true for 64-bit programs because "int" is still
 * 32 bits in size and "long" and "pointer" are now 64 bits (LP64)
 * so that the program crashes with a "Bus error" or something similar
 * if you store a pointer into an integer variable and run the program.
 * Therefore it is necessary to store the return value of a thread
 * into a pointer variable which holds a pointer if the thread returns
 * a pointer and which holds a value if it returns a value. You can
 * only return types which fit into a pointer variable, i. e., up to
 * 32-bit types in ILP32 and up to 64-bit types in LP64. Other types,
 * e. g., "long double", must be passed or returned via a pointer to
 * the variable which holds the value. With this in mind you can compile
 * your program in 32- and 64-bit mode. In certain circumstances it may
 * happen that the compiler produces an alignment error so that your
 * program will not get a correct return value. If you need a return
 * value from your thread and you want to compile your program in
 * 32- and 64-bit mode then the function should return a pointer to
 * something and not a basic data type.
 *
 *
 * Compiling:
 *   gcc -o <program name> <source code file name> -lm -lpthread
 *
 * Running:
 *   <program name>
 *
 *
 * File: dot_prod_block_pthread.c	Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define VECTOR_SIZE 100000000		/* vector size (10^8)		*/
#define NUM_THREADS 4			/* default number of threads	*/

/* Tests the return value of pthread functions.				*/
#define TestNotZero(val,file,line,function)  \
  if (val != 0) { fprintf (stderr, \
		  "File: %s, line %d: \"%s ()\" failed: %s\n", \
		  file, line, function, strerror (val)); \
		  exit (EXIT_FAILURE); }

/* global variables, so that all threads can easily access the values	*/
static double a[VECTOR_SIZE],		/* vectors for dot product	*/
	      b[VECTOR_SIZE];
static int    nthreads,			/* number of threads		*/
	      *my_first, *my_last;	/* first/last index for block	*/


double *compute_dot_product (int *thr_num);


int main (int argc, char *argv[])
{
  /* Use "thr_num" to avoid a warning about a cast to pointer from
   * integer of different size with "gcc -Wint-to-pointer-cast -m64".
   * Every thread needs its own variable, so that we need an array.
   * "partial_sum" is an array of pointers to double, so that each
   * thread can return its pointer to its own partial sum.
   */
  int *thr_num,				/* array of thread numbers	*/
      ret,				/* return value from functions	*/
      block_size,			/* min. block size for a thread	*/
      remainder;			/* # threads with bigger blocks	*/
  double sum, **partial_sum;
  pthread_t *mytid;			/* array of thread id's		*/
  pthread_attr_t attr;			/* thread attributes		*/

  /* evaluate command line arguments					*/
  switch (argc)
  {
    case 1:
      nthreads = NUM_THREADS;
      break;

    case 2:
      nthreads = atoi (argv[1]);
      if ((nthreads < 1) || (nthreads > VECTOR_SIZE))
      {
	fprintf (stderr, "Error: wrong number of threads.\n"
		 "Usage:\n"
		 "  %s number_of_threads\n"
		 "with 0 < number_of_threads <= %d\n"
		 "Use my default number of threads.\n",
		 argv[0], VECTOR_SIZE);
	nthreads = NUM_THREADS;
      }
      break;

    default:
      fprintf (stderr, "Error: wrong number of parameters.\n"
	       "Usage:\n"
	       "  %s number_of_threads\n"
	       "with 0 < number_of_threads <= %d\n",
	       argv[0], VECTOR_SIZE);
      exit (EXIT_FAILURE);
  }

  /* allocate memory for all dynamic data structures			*/
  my_first    = (int *) malloc ((size_t) nthreads * sizeof (int));
  my_last     = (int *) malloc ((size_t) nthreads * sizeof (int));
  thr_num     = (int *) malloc ((size_t) nthreads * sizeof (int));
  mytid       =
   (pthread_t *) malloc ((size_t) nthreads * sizeof (pthread_t));
  partial_sum =
   (double **) malloc ((size_t) nthreads * sizeof (double *));
  if ((my_first == NULL) || (my_last == NULL) || (thr_num == NULL) ||
      (mytid == NULL) || (partial_sum == NULL))
  {
    fprintf (stderr, "File: %s, line %d: Can't allocate memory.",
	     __FILE__, __LINE__);
    exit (EXIT_FAILURE);
  }

  /* initialize vectors							*/
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
    a[i] = 2.0;
    b[i] = 3.0;
  }

  /* Compute work for every thread. Some threads use a larger
   * block_size, if the number of threads isn't a factor of
   * VECTOR_SIZE.
   */
  block_size = VECTOR_SIZE / nthreads;
  remainder  = VECTOR_SIZE % nthreads;
  for (int i = 0; i < nthreads; ++i)
  {
    my_first[i] = (i < remainder) \
		    ? (i * (block_size + 1)) \
		    : (i * block_size + remainder);
    my_last[i]  = (i < remainder) \
		    ? ((i + 1) * (block_size + 1)) \
		    : ((i + 1) * block_size + remainder);
  }

  /* initialize thread objects						*/
  ret = pthread_attr_init (&attr);
  TestNotZero (ret, __FILE__, __LINE__, "pthread_attr_init");
  ret = pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_JOINABLE);
  TestNotZero (ret, __FILE__, __LINE__, "pthread_attr_setdetachstate");

  /* create threads							*/
  for (int i = 0; i < nthreads; ++i)
  {
    thr_num[i] = i;
    ret = pthread_create (&mytid[i], &attr,
			  (void * (*) (void *)) compute_dot_product,
			  (void *) &thr_num[i]);
    TestNotZero (ret, __FILE__, __LINE__, "pthread_create");
  }

  /* join threads and get result					*/
  for (int i = 0; i < nthreads; ++i)
  {
    ret = pthread_join (mytid[i], (void **) &partial_sum[i]);
    TestNotZero (ret, __FILE__, __LINE__, "pthread_join");
  }

  /* compute and print sum						*/
  sum = 0.0;
  for (int i = 0; i < nthreads; ++i)
  {
    sum += *partial_sum[i];
  }
  printf ("sum = %e\n", sum);
  
  /* clean up all things                                                */
  ret = pthread_attr_destroy (&attr);
  TestNotZero (ret, __FILE__, __LINE__, "pthread_attr_destroy");
  free (my_first);
  free (my_last);
  free (thr_num);
  free (mytid);
  for (int i = 0; i < nthreads; ++i)
  {
    free (partial_sum[i]);
  }
  free (partial_sum);
  return EXIT_SUCCESS;
}



/* compute_dot_product will be processed as a thread and returns
 * a pointer to its local sum of the dot product. The allocated
 * memory must be released in the caller.
 *
 * Input:		thr_num		ptr to current thread number
 * Output		none
 * Return value:	pointer to its local sum of dot product
 * Sideeffects:		none
 *
 */
double *compute_dot_product (int *thr_num)
{
  double *my_sum;

  my_sum = (double *) malloc (sizeof (double));
  if (my_sum == NULL)
  {
    fprintf (stderr, "File: %s, line %d: Can't allocate memory.",
	     __FILE__, __LINE__);
    exit (EXIT_FAILURE);
  }

  *my_sum = 0.0;
  for (int i = my_first[*thr_num]; i < my_last[*thr_num]; ++i)
  {
    #if VECTOR_SIZE < 20
      printf ("Thread %d: i = %d.\n", *thr_num, i);
    #endif
    *my_sum += a[i] * b[i];
  }
  pthread_exit ((void *) my_sum);
  /* The next statement isn't needed, but the compiler shouldn't
   * complain about a missing return-statement. The memory for
   * "my_sum" will be released at the end of function "main ()".
   */
  return my_sum;
}
