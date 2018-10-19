/* Simplified implementation of the DAXPY subprogram (double
 * precision alpha x plus y) from the Basic Linear Algebra
 * Subprogram library (BLAS). This subprogram computes
 * "y = alpha * x + y" with identical increments of size "1"
 * for the indexes of both vectors, so that we can omit the
 * increment parameters in the original function which has
 * the following prototype.
 *
 * void daxpy (int n, double alpha, double x[], int incx,
 *	       double y[], int incy);
 *
 *
 * File: daxpyKernel.cl			Author: S. Gross
 * Date: 29.12.2016
 *
 */


/* OpenCL kernel "daxpyKernel".
 *
 * input parameters:	n	number of elements in x and y
 *			alpha	scalar alpha for multiplication
 *			x	elements of vector x
 *			y	elements of vector y
 * output parameters:	y	updated elements of vector y
 * return value:	none
 * side effects:	elements of vector y will be overwritten
 *			  with new values
 *
 */

#if defined (cl_khr_fp64) || defined (cl_amd_fp64)
  __kernel void daxpyKernel (const int n,
			     const double alpha,
			     __global const double * restrict x,
			     __global double * restrict y)
  {
    for (int tid = get_global_id (0);
	 tid < n;
	 tid += get_global_size (0))
    {
      y[tid] += (alpha * x[tid]);
    }
  }
#else
  #error "Double precision floating point not supported."
#endif

