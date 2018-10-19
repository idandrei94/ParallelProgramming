/* This program displays the binding of threads to hardware threads,
 * cores, and/or sockets.
 *
 *
 * Compiling:
 *
 * cc  -xopenmp -o omp_affinity omp_affinity.c
 * gcc -fopenmp -o omp_affinity omp_affinity.c
 * icc -qopenmp -o omp_affinity omp_affinity.c
 * cl  /openmp omp_affinity.c
 *
 *
 * Running:
 *   Choose one value for the binding and another one for the places
 *   to bind to, for example:
 *
 *   setenv OMP_PROC_BIND true
 *   setenv OMP_PROC_BIND false
 *   setenv OMP_PROC_BIND master
 *   setenv OMP_PROC_BIND close
 *   setenv OMP_PROC_BIND spread
 *
 *   setenv OMP_PLACES threads
 *   setenv OMP_PLACES cores
 *   setenv OMP_PLACES sockets
 *
 *   Two 6-core processors with 2 hyperthreads / core
 *   setenv OMP_NUM_THREADS 1
 *   setenv OMP_NUM_THREADS 2
 *   setenv OMP_NUM_THREADS 6
 *   setenv OMP_NUM_THREADS 12
 *   setenv OMP_NUM_THREADS 24
 *
 *   ./omp_affinity
 *
 *
 * File: omp_affinity.c			Author: S. Gross
 * Date: 13.01.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

int main (void)
{
  /* OpenMP 4.5 (November 2015) is necessary				*/
  #if (_OPENMP >= 201511)
    #pragma omp parallel
    {
      int  ret,				/* return value			*/
	   max_bind,			/* # of entries in "bind_value"	*/
	   *place_nums,			/* list of place numbers	*/
	   num_places,			/* number of places		*/
	   part_num_places;		/* partition number of places	*/
      char *env_omp_places;		/* value of OMP_PLACES		*/
      char *bind_value[] = {"false", "true", "master", "close",
			    "spread"};
      omp_proc_bind_t ret_bind;		/* return value			*/

      num_places      = omp_get_num_places ();
      part_num_places = omp_get_partition_num_places ();
      max_bind	      = sizeof (bind_value) / sizeof (bind_value[0]);
      ret_bind	      = omp_get_proc_bind ();
      if (((int) ret_bind < 0) || ((int) ret_bind >= max_bind))
      {
	fprintf (stderr, "omp_get_proc_bind() returned wrong value\n");
      }
      env_omp_places = getenv ("OMP_PLACES");
      if (env_omp_places == NULL)
      {
	env_omp_places = "not set";
      }
      #pragma omp single
      {
	printf ("OMP_PROC_BIND                = %s\n"
		"OMP_PLACES                   = %s\n"
		"Number of places             = %d\n"
		"Number of partitions         = %d\n"
		"Number of places / partition = %d\n",
		bind_value[ret_bind], env_omp_places, num_places,
		(part_num_places > 0) ? num_places / part_num_places : 0,
		part_num_places);
      }

      /* all threads should print their messages sequentially		*/
      #pragma omp critical
      {
	/* print the list of place numbers of a partition		*/
	if (part_num_places > 0)
	{
	  place_nums = malloc (part_num_places * sizeof (int));
	  if (place_nums == NULL)
	  {
	    fprintf (stderr, "File: %s, line %d: Can't allocate "
		     "memory.\n", __FILE__, __LINE__);
	    exit (EXIT_FAILURE);
	  }
	  omp_get_partition_place_nums (place_nums);
	  printf ("\nThread %d: My partition contains the following "
		  "IDs:\n"
		  "  ", omp_get_thread_num ());
	  for (int j = 0; j < part_num_places - 1; ++j)
	  {
	    printf ("%d, ", place_nums[j]);
	  }
	  printf ("%d\n", place_nums[part_num_places - 1]);
	  free (place_nums);
	}

	ret = omp_get_place_num ();
	if (ret != -1)
	{
	  printf ("Thread %d is bound to place %d\n",
		  omp_get_thread_num (), ret);
	}
	else
	{
	  printf ("Thread %2d is not bound to a place\n",
		  omp_get_thread_num ());
	}
      }
    }
  #else
    printf ("OpenMP 4.5 (November 2015, 201511) or newer is necessary\n"
	    "for this program. Found only OpenMP %d.\n", _OPENMP);
  #endif
  return EXIT_SUCCESS;
}
