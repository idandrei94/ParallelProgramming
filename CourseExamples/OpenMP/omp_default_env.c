/* A small OpenMP program which prints the default/current values
 * of internal OpenMP variables.
 *
 *
 * Compiling:
 *
 * cc  -xopenmp -o omp_default_env omp_default_env.c
 * gcc -fopenmp -o omp_default_env omp_default_env.c
 * icc -qopenmp -o omp_default_env omp_default_env.c
 * cl  /openmp omp_default_env.c
 *
 *
 * Running:
 *   setenv OMP_NUM_THREADS 1		(csh, tcsh)
 *   ./omp_default_env
 *
 *
 * File: omp_default_env.c		Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
  #include <omp.h>
#else
  #define _OPENMP 201511		/* test with all stub functions	*/
  #include "omp_stubs.h"
#endif

static char *scheduling[]  = {"not defined", "static", "dynamic",
			      "guided", "auto"},
	    *bool_value[]  = {"false", "true"},
	    *bind_value[]  = {"false", "true", "master", "close",
			      "spread"};
static int  max_scheduling = sizeof (scheduling) / sizeof (scheduling[0]),
	    max_bool	   = sizeof (bool_value) / sizeof (bool_value[0]),
	    max_bind	   = sizeof (bind_value) / sizeof (bind_value[0]);

int main (void)
{
  int ret1, ret2, ret3, ret4, ret5;
  #if (_OPENMP >= 200805)
    int chunk_size;			/* param for omp_get_schedule()	*/
    omp_sched_t kind;			/* param for omp_get_schedule()	*/
  #endif
  #if (_OPENMP >= 201307)
    omp_proc_bind_t ret_bind;		/* for omp_get_proc_bind()	*/
  #endif

    printf ("\nFunctions since OpenMP 1.0.\n");
  /* functions in OpenMP 1.0 (October 1998)				*/
  #pragma omp parallel
  {
    #pragma omp single
    {
      ret1 = omp_get_num_procs ();
      ret2 = omp_get_num_threads ();
      ret3 = omp_get_max_threads ();
      ret4 = omp_get_dynamic ();
      ret5 = omp_get_nested ();
    }
  }
  if ((ret4 < 0) || (ret4 > max_bool))
  {
    fprintf (stderr, "omp_get_dynamic() returned wrong value\n");
    ret4 = 0;
  }
  if ((ret5 < 0) || (ret5 > max_bool))
  {
    fprintf (stderr, "omp_get_nested() returned wrong value\n");
    ret5 = 0;
  }
  printf ("_OPENMP                      = %d\n"
	  "omp_num_procs                = %d\n"
	  "omp_num_threads              = %d\n"
	  "omp_max_threads              = %d\n"
	  "omp_dynamic                  = %s\n"
	  "omp_nested                   = %s\n",
	  _OPENMP, ret1, ret2, ret3, bool_value[ret4],
	  bool_value[ret5]);

  /* new functions in OpenMP 3.0 (May 2008)				*/
  #if (_OPENMP >= 200805)
    printf ("\nNew functions in OpenMP 3.0.\n");
    #pragma omp parallel
    {
      #pragma omp single
      {
	ret1 = omp_get_level ();
	ret2 = omp_get_active_level ();
	ret3 = omp_get_max_active_levels ();
	ret4 = omp_get_thread_limit ();
	chunk_size = 0;
	kind	   = (omp_sched_t) 0;
	omp_get_schedule (&kind, &chunk_size);
      }
    }
    if (((int) kind < 0) || ((int) kind >= max_scheduling))
    {
      fprintf (stderr, "omp_get_schedule() returned wrong value\n");
      kind = (omp_sched_t) 0;
    }
    printf ("omp_level                    = %d\n"
	    "omp_active_level             = %d\n"
	    "omp_max_active_level         = %d\n"
	    "omp_thread_limit             = %d\n"
	    "omp_schedule                 = %s,%d\n",
	    ret1, ret2, ret3, ret4, scheduling[kind], chunk_size);
  #endif

  /* new functions in OpenMP 4.0 (July 2013)				*/
  #if (_OPENMP >= 201307)
    printf ("\nNew functions in OpenMP 4.0.\n");
    #pragma omp parallel
    {
      #pragma omp single
      {
	ret1 = omp_get_cancellation ();
	ret2 = omp_get_default_device ();
	ret3 = omp_get_num_devices ();
	ret4 = omp_get_num_teams ();
	ret5 = omp_get_team_num ();
	ret_bind = omp_get_proc_bind ();
      }
    }
    if ((ret1 < 0) || (ret1 > max_bool))
    {
      fprintf (stderr, "omp_get_cancellation() returned wrong "
	       "value\n");
      ret1 = 0;
    }
    if (((int) ret_bind < 0) || ((int) ret_bind >=  max_bind))
    {
      fprintf (stderr, "omp_get_proc_bind() returned wrong value\n");
      ret_bind = (omp_proc_bind_t) 0;
    }
    printf ("omp_cancellation             = %s\n"
	    "omp_default_device           = %d\n"
	    "omp_num_devices              = %d\n"
	    "omp_num_teams                = %d\n"
	    "omp_team_num                 = %d\n"
	    "omp_proc_bind                = %s\n",
	    bool_value[ret1], ret2, ret3, ret4, ret5,
	    bind_value[ret_bind]);
  #endif

  /* new functions in OpenMP 4.5 (November 2015)			*/
  #if (_OPENMP >= 201511)
    printf ("\nNew functions in OpenMP 4.5.\n");
    #pragma omp parallel
    {
      #pragma omp single
      {
	int *place_nums,		/* list of place numbers	*/
	    num_places,			/* number of places		*/
	    part_num_places;		/* partition number of places	*/

	ret1		= omp_get_initial_device ();
	ret2		= omp_get_max_task_priority ();
	num_places	= omp_get_num_places ();
	part_num_places = omp_get_partition_num_places ();
	printf ("omp_get_initial_device       = %d\n"
		"omp_get_max_task_priority    = %d\n"
		"omp_get_num_places           = %d\n"
		"omp_get_partition_num_places = %d\n",
		ret1, ret2, num_places, part_num_places);

	ret1 = omp_get_place_num ();
	if (ret1 != -1)
	{
	  printf ("Thread %d is bound to processor %d\n",
		  omp_get_thread_num (), ret1);
	}
	else
	{
	  printf ("Thread %d is not bound to a processor\n",
		  omp_get_thread_num ());
	}

	/* print numerical identifiers of the processors available
	 * to the execution environment for all places
	 */
	for (int i = 0; i < num_places; ++i)
	{
	  int *ids,			/* processor ids for place i	*/
	      num_procs;		/* # processors in place i	*/

	  num_procs = omp_get_place_num_procs (i);
	  if (num_procs > 0)
	  {
	    ids = malloc (num_procs * sizeof (int));
	    if (ids == NULL)
	    {
	      fprintf (stderr, "File: %s, line %d: Can't allocate "
		       "memory.\n", __FILE__, __LINE__);
	      exit (EXIT_FAILURE);
	    }
	    omp_get_place_proc_ids (i, ids);
	    printf ("Place list %d contains the following %d "
		    "processor ids:\n"
		    "  ", i, num_procs);
	    for (int j = 0; j < num_procs - 1; ++j)
	    {
	      printf ("%d, ", ids[j]);
	    }
	    printf ("%d\n", ids[num_procs - 1]);
	    free (ids);
	  }
	}

	/* print the list of place numbers corresponding to the
	 * places in the place-partition-var of the innermost
	 * implicit task
	 */
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
	  printf ("Partition contains the following %d "
		  "place numbers:\n"
		  "  ", part_num_places);
	  for (int j = 0; j < part_num_places - 1; ++j)
	  {
	    printf ("%d, ", place_nums[j]);
	  }
	  printf ("%d\n", place_nums[part_num_places - 1]);
	  free (place_nums);
	}
      }
    }
  #endif
  return EXIT_SUCCESS;
}
