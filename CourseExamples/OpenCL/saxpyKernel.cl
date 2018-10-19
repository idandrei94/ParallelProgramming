/* Simplified implementation of the DAXPY subprogram (double
 * precision alpha x plus y) from the Basic Linear Algebra
 * Subprogram library (BLAS). This subprogram computes
 * "y = alpha * x + y" with identical increments of size "1"
 * for the indexes of both vectors, so that we can omit the
 * increment parameters in the original function which has
 * the following prototype.
 *
 * void saxpy (int n, double alpha, double x[], int incx,
 *	       double y[], int incy);
 *
 *
 * File: saxpyKernel.cl			Author: S. Gross
 * Date: 29.12.2016
 *
 */


/* OpenCL kernel "saxpyKernel".
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

__kernel void saxpyKernel (const int n,
			   const float alpha,
			   __global const float * restrict x,
			   __global float * restrict y)
{
  for (int tid = get_global_id (0);
       tid < n;
       tid += get_global_size (0))
  {
    y[tid] += (alpha * x[tid]);
  }
}
