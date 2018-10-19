/* Header file for C/C++ stub routines from appendix B of the
 * standard "OpenMP Application Program Interface - Version 4.5
 * November 2015".
 *
 *
 * File:   omp_stubs.h
 *
 * Author: OpenMP Architecture Review Board
 *	   S. Gross: assembled data types and function prototypes
 *
 * Date:   22.03.2016
 *
 */

#ifndef _MY_OMP_STUBS_H
#define _MY_OMP_STUBS_H

/*
 * define the lock data types
 */
typedef void *omp_lock_t;
typedef void *omp_nest_lock_t;

/*
 * define the lock hints
 */
typedef enum omp_lock_hint_t {		/* new in OpenMP 4.5		*/
  omp_lock_hint_none = 0,
  omp_lock_hint_uncontended = 1,
  omp_lock_hint_contended = 2,
  omp_lock_hint_nonspeculative = 4,
  omp_lock_hint_speculative = 8
  /* Add vendor specific constants for lock hints here,
   * starting from the most-significant bit.
   */
} omp_lock_hint_t;

/*
 * define the schedule kinds
 */
typedef enum omp_sched_t {		/* new in OpenMP 3.0		*/
  omp_sched_static = 1,
  omp_sched_dynamic = 2,
  omp_sched_guided = 3,
  omp_sched_auto = 4
  /* Add vendor specific schedule constants here			*/
} omp_sched_t;

/*
 * define the proc bind values
 */
typedef enum omp_proc_bind_t {		/* new in OpenMP 4.0		*/
  omp_proc_bind_false = 0,
  omp_proc_bind_true = 1,
  omp_proc_bind_master = 2,
  omp_proc_bind_close = 3,
  omp_proc_bind_spread = 4
} omp_proc_bind_t;


/*
 * exported OpenMP functions
 */
#ifdef __cplusplus
extern "C"
{
#endif

/* functions in OpenMP 1.0 (October 1998)				*/
extern int omp_get_dynamic (void);
extern int omp_get_nested (void);
extern int omp_get_num_procs (void);
extern int omp_get_num_threads (void);
extern int omp_get_max_threads (void);
extern int omp_get_thread_num (void);
extern int omp_in_parallel (void);
extern void omp_set_dynamic (int dynamic_threads);
extern void omp_set_nested (int nested);
extern void omp_set_num_threads (int num_threads);

extern void omp_destroy_lock (omp_lock_t *lock);
extern void omp_destroy_nest_lock (omp_nest_lock_t *lock);
extern void omp_init_lock (omp_lock_t *lock);
extern void omp_init_nest_lock (omp_nest_lock_t *lock);
extern void omp_set_lock (omp_lock_t *lock);
extern void omp_set_nest_lock (omp_nest_lock_t *lock);
extern int omp_test_lock (omp_lock_t *lock);
extern int omp_test_nest_lock (omp_nest_lock_t *lock);
extern void omp_unset_lock (omp_lock_t *lock);
extern void omp_unset_nest_lock (omp_nest_lock_t *lock);

/* new functions in OpenMP 2.0 (March 2002)				*/
extern double omp_get_wtick (void);
extern double omp_get_wtime (void);

/* no new functions in OpenMP 2.5 (May 2005)				*/

/* new functions in OpenMP 3.0 (May 2008)				*/
extern int omp_get_active_level (void);
extern int omp_get_ancestor_thread_num (int level);
extern int omp_get_level (void);
extern int omp_get_max_active_levels (void);
extern void omp_get_schedule (omp_sched_t *kind, int *chunk_size);
extern int omp_get_team_size (int level);
extern int omp_get_thread_limit (void);
extern void omp_set_max_active_levels (int max_active_levels);
extern void omp_set_schedule (omp_sched_t kind, int chunk_size);

/* new functions in OpenMP 3.1 (July 2011)				*/
extern int omp_in_final (void);

/* new functions in OpenMP 4.0 (July 2013)				*/
extern int omp_get_cancellation (void);
extern int omp_get_default_device (void);
extern int omp_get_num_devices (void);
extern int omp_get_num_teams (void);
extern omp_proc_bind_t omp_get_proc_bind (void);
extern int omp_get_team_num (void);
extern int omp_is_initial_device (void);
extern void omp_set_default_device (int device_num);

/* new functions in OpenMP 4.5 (November 2015)				*/
extern int omp_get_initial_device (void);
extern int omp_get_max_task_priority (void);
extern int omp_get_num_places (void);
extern int omp_get_partition_num_places (void);
extern void omp_get_partition_place_nums (int *place_nums);
extern int omp_get_place_num (void);
extern int omp_get_place_num_procs (int place_num);
extern void omp_get_place_proc_ids (int place_num, int *ids);

extern void omp_init_lock_with_hint (omp_lock_t *lock,
				     omp_lock_hint_t hint);
extern void omp_init_nest_lock_with_hint (omp_nest_lock_t *lock,
					  omp_lock_hint_t hint);


extern void *omp_target_alloc (size_t size, int device_num);
extern int omp_target_associate_ptr (void *host_ptr,
				     void *device_ptr,
				     size_t size,
				     size_t device_offset,
				     int device_num);
extern int omp_target_disassociate_ptr (void *ptr,
					int device_num);
extern void omp_target_free (void *device_ptr, int device_num);
extern int omp_target_is_present (void *ptr, int device_num);
extern int omp_target_memcpy (void *dst, void *src, size_t length,
			      size_t dst_offset, size_t src_offset,
			      int dst_device_num, int src_device_num);
extern int omp_target_memcpy_rect (void *dst, void *src,
				   size_t element_size,
				   int num_dims,
				   const size_t *volume,
				   const size_t *dst_offsets,
				   const size_t *src_offsets,
				   const size_t *dst_dimensions,
				   const size_t *src_dimensions,
				   int dst_device_num, int src_device_num);


#ifdef __cplusplus
}
#endif

#endif
