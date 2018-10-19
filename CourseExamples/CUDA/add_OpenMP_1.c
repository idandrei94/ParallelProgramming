/* A simple program adding two vectors without a GPU, but with
 * parallel code (OpenMP) for "DEFAULT_NUM_THREADS" threads with
 * static scheduling and a chunk size of "1", which is equivalent
 * to the work of one CUDA thread.
 *
 *
 * Compiling:
 *   gcc -fopenmp -o add_OpenMP_1 add_OpenMP_1.c
 *   clang -Xcompiler -fopenmp -o add_OpenMP_1 add_OpenMP_1.c
 *
 * Running:
 *   ./add_OpenMP_1 [vector size]
 *
 *
 * File: add_OpenMP_1.c			Author: S. Gross
 * Date: 03.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#ifdef _OPENMP
  #include <omp.h>
#endif


/* Microsoft Visual Studio (at least up to version 2017) doesn't
 * support the keyword "restrict" for pointers.
 */
#if defined(WIN32) || defined(_WIN32) || defined(Win32)
  #define RESTRICT 
#else 
  #define RESTRICT restrict
#endif 

#define DEFAULT_VECTOR_SIZE  100000000	/* vector size (10^8)		*/
#define DEFAULT_NUM_THREADS  4		/* number of threads to use	*/


void vecAdd (const int *RESTRICT a, const int *RESTRICT b,
	     int *RESTRICT c, const size_t vecSize,
	     const int threadID, const int numThreads);
int  checkResult (const int *RESTRICT a, const int *RESTRICT b,
		  const int *RESTRICT c, const size_t vecSize);
void evalCmdLine (int argc, char *argv[], size_t *vecSize);


/* define macro to test the result of a "malloc" operation		*/
#define TestEqualsNULL(val)  \
  if (val == NULL) \
  { \
    fprintf (stderr, "file: %s  line %d: Couldn't allocate memory.\n", \
	     __FILE__, __LINE__); \
    exit (EXIT_FAILURE); \
  }


int main (int argc, char *argv[])
{
  int	  *a, *b, *c,			/* vector addresses		*/
	  i,				/* loop variable		*/
	  result;			/* result of comparison		*/
  size_t  vecSize;			/* vector size			*/
  time_t  start_wall, end_wall,		/* start/end time (wall clock)	*/
	  start_total_wall,
    	  end_total_wall;
  clock_t cpu_time;			/* used cpu time		*/

  /* check for command line argument					*/
  evalCmdLine (argc, argv, &vecSize);

  /* measure the total wall clock time					*/
  start_total_wall = time (NULL);

  /* allocate memory for all vectors					*/
  printf ("Allocate memory on CPU.\n");
  a = (int *) malloc (vecSize * sizeof (int));
  TestEqualsNULL (a);
  b = (int *) malloc (vecSize * sizeof (int));
  TestEqualsNULL (b);
  c = (int *) malloc (vecSize * sizeof (int));
  TestEqualsNULL (c);

  /* initialize vectors							*/
  printf ("Initializing arrays.\n");
  for (i = 0; i < (int) vecSize; ++i)
  {
    a[i] = i;
    b[i] = i;
    c[i] = 0;
  }

  /* add vectors and measure computation time				*/
  printf ("Adding arrays.\n");
  start_wall = time (NULL);
  cpu_time = clock ();
  #pragma omp parallel for default(none) shared(a, b, c, vecSize) \
    schedule(static, 1) num_threads(DEFAULT_NUM_THREADS)
  for (i = 0; i < DEFAULT_NUM_THREADS; ++i)
  {
    vecAdd (a, b, c, vecSize, i, DEFAULT_NUM_THREADS);
  }
  cpu_time = clock () - cpu_time;
  end_wall = time (NULL);

  /* check result and clean up						*/
  printf ("Checking result.\n");
  result = checkResult (a, b, c, vecSize);

  printf ("Cleaning up.\n");
  free (a);
  free (b);
  free (c);
  end_total_wall = time (NULL);

  /* show all times							*/
  printf ("\nelapsed time      cpu time      total wall clock time\n"
	  "    %6.2f s      %6.2f s                   %6.2f s\n",
	  difftime (end_wall, start_wall),
	  (double) cpu_time / CLOCKS_PER_SEC,
	  difftime (end_total_wall, start_total_wall));
  return result;
}


/* Add vectors "a" and "b" and store the result into vector "c".
 *
 * Input:		a, b		vectors to be added
 *			vecSize		size of all vectors
 *			threadID	starting index for vectors
 *			numThreads	number of threads
 * Output		c		result vector
 * Return value:	none
 * Sideeffects:		none
 *
 */
void vecAdd (const int *RESTRICT a, const int *RESTRICT b,
	     int *RESTRICT c, const size_t vecSize,
	     const int threadID, const int numThreads)
{
  int i;				/* loop variable		*/

  for (i = threadID; i < (int) vecSize; i += numThreads)
  {
    c[i] = a[i] + b[i];
  }
}


/* Compare the sum of "a" and "b" with the values of "c".
 *
 * Input:		a, b		vectors to be added
 *			c		original result vector
 *			vecSize		size of all vectors
 * Output		none
 * Return value:	EXIT_SUCCESS	if c == a + b
 *			EXIT_FAILURE	if c != a + b
 * Sideeffects:		none
 *
 */
int  checkResult (const int *RESTRICT a, const int *RESTRICT b,
		  const int *RESTRICT c, const size_t vecSize)
{
  int i,				/* loop variable		*/
      result = EXIT_SUCCESS;

  for (i = 0; (i < (int) vecSize) && (result == EXIT_SUCCESS); ++i)
  {
    if (c[i] != a[i] + b[i])
    {
      result = EXIT_FAILURE;
    }	
  }
  if (result == EXIT_SUCCESS)
  {
    printf ("Adding two vectors completed successfully.\n");
  }
  else
  {
    printf ("Adding two vectors failed.\n");
  }

  return result;
}


/* Evaluate command line arguments and set the vector size to a
 * default value or a value requested on the command line.
 *
 * Input:		argc		argument count
 *			argv		argument vector
 * Output		vecSize		vector size
 * Return value:	none
 * Sideeffects:		terminates the program after printing a
 *			help message, if the command line contains
 *			too many arguments
 *
 */
void evalCmdLine (int argc, char *argv[], size_t *vecSize)
{
  switch (argc)
  {
    case 1:				/* no parameters on cmd line	*/
      *vecSize = DEFAULT_VECTOR_SIZE;
      break;

    case 2:				/* one parameter on cmd line	*/
      *vecSize = (size_t) atoi (argv[1]);
      break;

    default:
      fprintf (stderr, "\nError: too many parameters.\n"
	       "Usage: %s [size of vector]\n\n", argv[0]);
      exit (EXIT_FAILURE);
  }

  /* ensure that all values are valid					*/
  if (*vecSize < 1)
  {
    fprintf (stderr, "\nError: Vector size must be greater than zero.\n"
	     "I use the default size: %d.\n\n", DEFAULT_VECTOR_SIZE);
    *vecSize = DEFAULT_VECTOR_SIZE;
  }
}
