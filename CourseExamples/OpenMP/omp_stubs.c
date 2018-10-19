/* C/C++ stub routines from appendix A of the standard "OpenMP
 * Application Program Interface - Version 4.5 November 2015"
 *
 * "_POSIX_C_SOURCE" and _XOPEN_SOURCE must be defined before any
 * "#include".
 * The value for this macro depends on compiler options (ISO-C99
 * compatability, ANSI-C compatability, ...). You can use the
 * command "man standards" to learn about possible values for Linux
 * and Solaris/SunOS. For Linux you can also use the command
 * "man feature_test_macros". Furthermore you can use the command
 * "more /usr/include/features.h" for Linux and Cygwin, the command
 * "more /usr/include/sys/cdefs.h" for Mac OS X (Darwin) and the
 * command "more /usr/include/sys/feature_tests.h" for Solaris to
 * learn about all possible feature test macros.
 *
 *
 * File:   omp_stubs.c
 *
 * Author: OpenMP Architecture Review Board
 *	   S. Gross: rearranging and adding content of omp_get_time()
 *		     and omp_get_wtick()
 *
 * Date:   14.10.2016
 *
 */

#ifdef ANSI_C
  #define _POSIX_C_SOURCE 199506L	/* standard: June 1995		*/
  #define _XOPEN_SOURCE   500
#else
  #if defined(SunOS)
    #define _POSIX_C_SOURCE 200112L	/* standard: December 2001	*/
    #define _XOPEN_SOURCE   600
  #else
    #define _POSIX_C_SOURCE 200809L	/* standard: September 2008	*/
    #define _XOPEN_SOURCE   700
  #endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#if defined(SunOS) || defined(Linux) || defined(Cygwin)
  #include <sys/errno.h>
  #include <sys/types.h>
  #include <time.h>
  #include <string.h>
  #if !defined(CLOCK_MONOTONIC)
    #define CLOCK_MONOTONIC (clockid_t) -1	/* Cygwin 1.5.x		*/
  #endif
#endif
#if defined(Darwin) || defined(__APPLE__)
  #include <sys/errno.h>
  #include <sys/types.h>
  #include <time.h>
  #include <mach/clock.h>
  #include <mach/mach.h>
#endif
#ifdef Win32
  #include <errno.h>
  #include <windows.h>
#endif

#include "omp_stubs.h"


/* some variables to throw away unused parameters so that "gcc"
 * will not warn about "unused parameter ..."
 */
int throw_away_int_val;
omp_lock_hint_t throw_away_omp_lock_hint_t_val;
omp_sched_t throw_away_omp_sched_t_val;
size_t throw_away_size_t_val;
void *throw_away_void_ptr_val;


/* some internal data structures and constants
 *
 */
struct __omp_lock
{
  int lock;
};

struct __omp_nest_lock
{
  short owner;
  short count;
};

enum { UNLOCKED = -1, INIT, LOCKED };	/* __omp_lock			*/
enum { NOOWNER = -1, MASTER = 0 };	/* __omp_nest_lock		*/


/* stub functions
 *
 */
void omp_set_num_threads (int num_threads)
{
  throw_away_int_val = num_threads;
}


int omp_get_num_threads (void)
{
  return 1;
}


int omp_get_max_threads (void)
{
  return 1;
}


int omp_get_thread_num (void)
{
  return 0;
}


int omp_get_num_procs (void)
{
  return 1;
}

int omp_in_parallel (void)
{
  return 0;
}


void omp_set_dynamic (int dynamic_threads)
{
  throw_away_int_val = dynamic_threads;
}


int omp_get_dynamic (void)
{
  return 0;
}


void omp_set_nested (int nested)
{
  throw_away_int_val = nested;
}


int omp_get_nested (void)
{
  return 0;
}


void omp_set_schedule (omp_sched_t kind, int chunk_size)
{
  throw_away_omp_sched_t_val = kind;
  throw_away_int_val	     = chunk_size;
}


void omp_get_schedule (omp_sched_t *kind, int *chunk_size)
{
  *kind	      = omp_sched_static;
  *chunk_size = 0;
}


int omp_get_thread_limit (void)
{
  return 1;
}


void omp_set_max_active_levels (int max_active_levels)
{
  throw_away_int_val = max_active_levels;
}


int omp_get_max_active_levels (void)
{
  return 0;
}


int omp_get_level (void)
{
  return 0;
}


int omp_get_ancestor_thread_num (int level)
{
  if (level == 0)
  {
    return 0;
  }
  else
  {
    return -1;
  }
}


int omp_get_team_size (int level)
{
  if (level == 0)
  {
    return 1;
  }
  else
  {
    return -1;
  }
}


int omp_get_active_level (void)
{
  return 0;
}


int omp_in_final (void)
{
  return 1;
}


void omp_init_lock (omp_lock_t *arg)
{
  struct __omp_lock *lock = (struct __omp_lock *) arg;

  lock->lock = UNLOCKED;
}


void omp_destroy_lock (omp_lock_t *arg)
{
  struct __omp_lock *lock = (struct __omp_lock *) arg;

  lock->lock = INIT;
}


void omp_set_lock (omp_lock_t *arg)
{
  struct __omp_lock *lock = (struct __omp_lock *) arg;

  if (lock->lock == UNLOCKED)
  {
    lock->lock = LOCKED;
  }
  else
  {
    if (lock->lock == LOCKED)
    {
      fprintf (stderr, "error: deadlock in using lock variable\n");
      exit (1);
    }
    else
    {
      fprintf (stderr, "error: lock not initialized\n");
      exit (1);
    }
  }
}


void omp_unset_lock (omp_lock_t *arg)
{
  struct __omp_lock *lock = (struct __omp_lock *) arg;

  if (lock->lock == LOCKED)
  {
    lock->lock = UNLOCKED;
  }
  else
  {
    if (lock->lock == UNLOCKED)
    {
      fprintf (stderr, "error: lock not set\n");
      exit (1);
    }
    else
    {
      fprintf (stderr, "error: lock not initialized\n");
      exit (1);
    }
  }
}


int omp_test_lock (omp_lock_t *arg)
{
  int ret;

  struct __omp_lock *lock = (struct __omp_lock *) arg;

  if (lock->lock == UNLOCKED)
  {
    lock->lock = LOCKED;
    ret = 1;
  }
  else
  {
    if (lock->lock == LOCKED)
    {
      ret = 0;
    }
    else
    {
      fprintf (stderr, "error: lock not initialized\n");
      exit (1);
    }
  }
  return ret;
}


void omp_init_nest_lock (omp_nest_lock_t *arg)
{
  struct __omp_nest_lock *nlock = (struct __omp_nest_lock *) arg;

  nlock->owner = NOOWNER;
  nlock->count = 0;
}


void omp_destroy_nest_lock (omp_nest_lock_t *arg)
{
  struct __omp_nest_lock *nlock = (struct __omp_nest_lock *) arg;

  nlock->owner = NOOWNER;
  nlock->count = UNLOCKED;
}


void omp_set_nest_lock (omp_nest_lock_t *arg)
{
  struct __omp_nest_lock *nlock = (struct __omp_nest_lock *) arg;

  if ((nlock->owner == MASTER) && (nlock->count >= 1))
  {
    nlock->count++;
  }
  else
  {
    if ((nlock->owner == NOOWNER) && (nlock->count == 0))
    {
      nlock->owner = MASTER;
      nlock->count = 1;
    }
    else
    {
      fprintf (stderr, "error: lock corrupted or not initialized\n");
      exit (1);
    }
  }
}


void omp_unset_nest_lock (omp_nest_lock_t *arg)
{
  struct __omp_nest_lock *nlock = (struct __omp_nest_lock *) arg;

  if ((nlock->owner == MASTER) && (nlock->count >= 1))
  {
    nlock->count--;
    if (nlock->count == 0)
    {
      nlock->owner = NOOWNER;
    }
  }
  else
  {
    if ((nlock->owner == NOOWNER) && (nlock->count == 0))
    {
      fprintf (stderr, "error: lock not set\n");
      exit (1);
    }
    else
    {
      fprintf (stderr, "error: lock corrupted or not initialized\n");
      exit (1);
    }
  }
}


int omp_test_nest_lock (omp_nest_lock_t *arg)
{
  struct __omp_nest_lock *nlock = (struct __omp_nest_lock *) arg;

  omp_set_nest_lock (arg);
  return nlock->count;
}


/* Documentation for Darwin:
 *   https://developer.apple.com/library/mac/documentation/Darwin/
 *     Conceptual/KernelProgramming/About/About.html
 *   http://www.darwin-development.org/cgi-bin/cvsweb/osfmk/
 *     src/mach_kernel
 *   http://web.mit.edu/darwin/src/modules/xnu/osfmk/man
 *   http://www.gnu.org/software/hurd/gnumach-doc/index.html
 *
 * Documentation for Windows:
 *   http://msdn.microsoft.com/en-us/library/windows/desktop/ms724953.aspx
 */
double omp_get_wtime (void)
{
#if defined(SunOS) || defined(Linux) || defined(Cygwin)
  struct timespec ts;
  int	 ret;

  if (clock_gettime (CLOCK_MONOTONIC, &ts) < 0)
  {
    if ((ret = clock_gettime (CLOCK_REALTIME, &ts)) < 0)
    {
      fprintf (stderr, "\nError:  Couldn't get time.\n"
	       "Reason: %s\n", strerror (ret));
      ts.tv_sec  = 0;
      ts.tv_nsec = 0;
    }
  }
  return ts.tv_sec + (ts.tv_nsec / 1e9);
#endif

#ifdef Darwin
  mach_timespec_t ts;
  clock_serv_t	  clock_port;
  int		  ret;

  /* SYSTEM_CLOCK and CALENDAR_CLOCK have the same resolution, but
   * SYSTEM_CLOCK returns the elapsed time since the last system boot
   * and CALENDAR_CLOCK the epoch time (since 1970/01/01)
   */
  ret = host_get_clock_service (mach_host_self (), CALENDAR_CLOCK,
				&clock_port);
  if (ret != KERN_SUCCESS)
  {
    fprintf (stderr, "\nError:  Couldn't get send right to clock port.\n"
	     "Reason: %s\n", mach_error_string (ret));
  }
  ret = clock_get_time (clock_port, &ts);
  if (ret != KERN_SUCCESS)
  {
    fprintf (stderr, "\nError:  Couldn't get time.\n"
	     "Reason: %s\n", mach_error_string (ret));
  }
  ret = mach_port_deallocate (mach_task_self (), clock_port);
  if (ret != KERN_SUCCESS)
  {
    fprintf (stderr, "\nError:  Couldn't release clock port.\n"
	     "Reason: %s\n", mach_error_string (ret));
  }
  return ts.tv_sec + ts.tv_nsec / 1e9;
#endif

#ifdef Win32
  LONGLONG frequency, count;

  if (!QueryPerformanceFrequency ((LARGE_INTEGER *) &frequency))
  {
    fprintf (stderr, "\nError: performance counter not available.\n"
	     "Cannot return the current time.\n\n");
    return 0.0;
  }
  QueryPerformanceCounter ((LARGE_INTEGER *) &count);
  return (double) (count / frequency);
#endif

  /* the following statements should never be reached			*/
  fprintf (stderr, "\n\nError: Missing alternative in "
	   "\"omp_get_wtimer()\"?.\n"
	   "Perhaps you forgot to specify the operating system when\n"
	   "you compiled the program. Please compile once more and\n"
	   "specify -DCygwin, -DDarwin, -DLinux, -DSunOS, or /Win32\n"
	   "when you compile the program. If you use a different\n"
	   "operating system you have to extend this function.\n"
	   "Cannot return the current time.\n");
  return 0.0;
}


/* Documentation for Darwin:
 *   https://developer.apple.com/library/mac/documentation/Darwin/
 *     Conceptual/KernelProgramming/About/About.html
 *   http://www.darwin-development.org/cgi-bin/cvsweb/osfmk/
 *     src/mach_kernel
 *   http://web.mit.edu/darwin/src/modules/xnu/osfmk/man
 *   http://www.gnu.org/software/hurd/gnumach-doc/index.html
 *
 * Documentation for Windows:
 *   http://msdn.microsoft.com/en-us/library/windows/desktop/ms724953.aspx
 */
double omp_get_wtick (void)
{
#if defined(SunOS) || defined(Linux) || defined(Cygwin)
  struct timespec ts;
  int	 ret;

  if (clock_getres (CLOCK_MONOTONIC, &ts) < 0)
  {
    if ((ret = clock_getres (CLOCK_REALTIME, &ts)) < 0)
    {
      fprintf (stderr, "\nError:  Couldn't get clock resolution.\n"
	       "Reason: %s\n", strerror (ret));
      ts.tv_sec  = 0;
      ts.tv_nsec = 0;
    }
  }
  return ts.tv_sec + (ts.tv_nsec / 1e9);
#endif

#ifdef Darwin
  mach_msg_type_number_t attr_count;
  clock_serv_t		 clock_port;
  int			 nano_secs, ret;

  /* SYSTEM_CLOCK and CALENDAR_CLOCK have the same resolution, but
   * SYSTEM_CLOCK returns the elapsed time since the last system boot
   * and CALENDAR_CLOCK the epoch time (since 1970/01/01)
   */
  ret = host_get_clock_service (mach_host_self (), CALENDAR_CLOCK,
				&clock_port);
  if (ret != KERN_SUCCESS)
  {
    fprintf (stderr, "\nError:  Couldn't get send right to clock port.\n"
	     "Reason: %s\n", mach_error_string (ret));
  }
  attr_count = 1;
  ret = clock_get_attributes (clock_port, CLOCK_GET_TIME_RES,
			      (clock_attr_t) &nano_secs, &attr_count);
  if (ret != KERN_SUCCESS)
  {
    fprintf (stderr, "\nError:  Couldn't get clock resolution.\n"
	     "Reason: %s\n", mach_error_string (ret));
    nano_secs = 0;
  }
  ret = mach_port_deallocate (mach_task_self (), clock_port);
  if (ret != KERN_SUCCESS)
  {
    fprintf (stderr, "\nError:  Couldn't release clock port.\n"
	     "Reason: %s\n", mach_error_string (ret));
  }
  return (double) (nano_secs / 1e9);
#endif

#ifdef Win32
  LONGLONG frequency;

  if (!QueryPerformanceFrequency ((LARGE_INTEGER *) &frequency))
  {
    fprintf (stderr, "\nError: performance counter not available.\n"
	     "Cannot return timer resolution.\n\n");
    return 1.0;
  }
  else
  {
    return (double) (1.0 / frequency);
  }
#endif

  /* the following statements should never be reached			*/
  fprintf (stderr, "\n\nError: Missing alternative in "
	   "\"omp_get_wtick()\"?.\n"
	   "Perhaps you forgot to specify the operating system when\n"
	   "you compiled the program. Please compile once more and\n"
	   "specify -DCygwin, -DDarwin, -DLinux, -DSunOS, or /Win32\n"
	   "when you compile the program. If you use a different\n"
	   "operating system you have to extend this function.\n"
	   "Cannot return timer resolution.\n");
  return 1.0;
}



/* new functions in OpenMP 4.0 (July 2013)
 *
 */

int omp_get_cancellation (void)
{
  return 0;
}


omp_proc_bind_t omp_get_proc_bind (void)
{
  return omp_proc_bind_false;
}


void omp_set_default_device (int device_num)
{
  throw_away_int_val = device_num;
}


int omp_get_default_device (void)
{
  return 0;
}


int omp_get_num_devices (void)
{
  return 0;
}


int omp_get_num_teams (void)
{
  return 1;
}


int omp_get_team_num (void)
{
  return 0;
}


int omp_is_initial_device (void)
{
  return 1;
}



/* new functions in OpenMP 4.5 (November 2015)
 *
 */
int omp_get_initial_device (void)
{
  return -10;
}


int omp_get_max_task_priority (void)
{
  return 0;
}


int omp_get_num_places (void)
{
  return 0;
}


int omp_get_partition_num_places (void)
{
  return 0;
}


void omp_get_partition_place_nums (int *place_nums)
{
  throw_away_void_ptr_val = (void *) place_nums;
}


int omp_get_place_num (void)
{
  return -1;
}


int omp_get_place_num_procs (int place_num)
{
  throw_away_int_val = place_num;
  return 0;
}


void omp_get_place_proc_ids (int place_num, int *ids)
{
  throw_away_int_val	  = place_num;
  throw_away_void_ptr_val = (void *) ids;
}


void omp_init_lock_with_hint (omp_lock_t *arg, omp_lock_hint_t hint)
{
  throw_away_omp_lock_hint_t_val = hint;
  omp_init_lock (arg);
}


void omp_init_nest_lock_with_hint (omp_nest_lock_t *arg,
				   omp_lock_hint_t hint)
{
  throw_away_omp_lock_hint_t_val = hint;
  omp_init_nest_lock (arg);
}


void *omp_target_alloc (size_t size, int device_num)
{
  if (device_num != -10)
  {
    return NULL;
  }
  else
  {
    return malloc (size);
  }
}


int omp_target_associate_ptr (void *host_ptr, void *device_ptr,
			      size_t size, size_t device_offset,
			      int device_num)
{
  throw_away_void_ptr_val = host_ptr;
  throw_away_void_ptr_val = device_ptr;
  throw_away_size_t_val = size;
  throw_away_size_t_val = device_offset;
  throw_away_int_val = device_num;
  /* No association is possible because all host pointers
   * are considered present
   */
  return EINVAL;
}


int omp_target_disassociate_ptr (void *ptr, int device_num)
{
  throw_away_void_ptr_val = ptr;
  throw_away_int_val = device_num;
  return EINVAL;
}


void omp_target_free (void *device_ptr, int device_num)
{
  throw_away_int_val = device_num;
  free (device_ptr);
}


int omp_target_is_present (void *ptr, int device_num)
{
  throw_away_void_ptr_val = ptr;
  throw_away_int_val = device_num;
  return 1;
}


int omp_target_memcpy (void *dst, void *src, size_t length,
		       size_t dst_offset, size_t src_offset,
		       int dst_device, int src_device)
{
  /* only the default device is valid in a stub				*/
  if ((dst_device != -10) || (src_device != -10) ||
      (dst == NULL) || (src == NULL))
  {
    return EINVAL;
  }
  memcpy ((char *) dst + dst_offset, (char *) src + src_offset, length);
  return 0;
}


int omp_target_memcpy_rect (void *dst, void *src,
			    size_t element_size, int num_dims,
			    const size_t *volume,
			    const size_t *dst_offsets,
			    const size_t *src_offsets,
			    const size_t *dst_dimensions,
			    const size_t *src_dimensions,
			    int dst_device_num, int src_device_num)
{
  int ret = 0;

  /* Both null, return number of dimensions supported.
   * This stub supports an arbitrary number.
   */
  if ((dst == NULL) && (src == NULL))
  {
    return INT_MAX;
  }

  if ((volume == NULL) || (dst_offsets == NULL) ||
      (src_offsets == NULL) || (dst_dimensions == NULL) ||
      (src_dimensions == NULL) || (num_dims < 1))
  {
    return EINVAL;
  }
  if (num_dims == 1)
  {
    ret = omp_target_memcpy (dst, src,
			     volume[0] * element_size,
			     dst_offsets[0] * element_size,
			     src_offsets[0] * element_size,
			     dst_device_num, src_device_num);
  }
  else
  {
    size_t dst_slice_size = element_size;
    size_t src_slice_size = element_size;
    size_t dst_off;
    size_t src_off;

    for (int i = 1; i < num_dims; i++)
    {
      dst_slice_size *= dst_dimensions[i];
      src_slice_size *= src_dimensions[i];
    }
    dst_off = dst_offsets[0] * dst_slice_size;
    src_off = src_offsets[0] * src_slice_size;
    for (size_t i = 0; i < volume[0]; i++)
    {
      ret = omp_target_memcpy_rect (
		(char *) dst + dst_off + dst_slice_size * i,
		(char *) src + src_off + src_slice_size * i,
		element_size,
		num_dims - 1,
		volume + 1,
		dst_offsets + 1,
		src_offsets + 1,
		dst_dimensions + 1,
		src_dimensions + 1,
		dst_device_num,
		src_device_num);
      if (ret != 0)
      {
	return ret;
      }
    }
  }
  return ret;
}
