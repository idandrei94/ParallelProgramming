/* Kernel to compute the dot product.
 *
 *
 * File: dotProdKernel.cl		Author: S. Gross
 * Date: 04.01.2017
 *
 */

/* Compute dot product of vectors "a" and "b" and store partial
 * results of the work-items of each work-group into "partial_sum".
 *
 * Input:		a, b		vectors for dot product
 * Output		partial_sum	partial sums of a work-group
 * Return value:	none
 * Sideeffects:		none
 *
 */
#if defined (cl_khr_fp64) || defined (cl_amd_fp64)
  #include "dot_prod_OpenCL.h"

  __kernel void dotProdKernel (__global const double * restrict a,
			       __global const double * restrict b,
			       __global double * restrict partial_sum)
  {
    /* Use local memory to store each work-items running sum.		*/
    __local double cache[WORK_ITEMS_PER_WORK_GROUP];

    double temp = 0.0;
    int    cacheIdx = get_local_id (0);

    for (int tid = get_global_id (0);
	 tid < VECTOR_SIZE;
	 tid += get_global_size (0))
    {
      temp += a[tid] * b[tid];
    }
    cache[cacheIdx] = temp;

    /* Ensure that all work-items have completed, before you add up the
     * partial sums of each work-item to the sum of the work-group
     */
    barrier (CLK_LOCAL_MEM_FENCE);

    /* Each work-item will add two values and store the result back to
     * "cache". We need "log_2 (WORK_ITEMS_PER_WORK_GROUP)" steps to
     * reduce all partial values to one work-group value.
     * WORK_ITEMS_PER_WORK_GROUP must be a power of two for this
     * reduction.
     */
    for (int i = get_local_size (0) / 2; i > 0; i /= 2)
    {
      if (cacheIdx < i)
      {
	cache[cacheIdx] += cache[cacheIdx + i];
      }
      barrier (CLK_LOCAL_MEM_FENCE);
    }
    /* store the partial sum of this work-group				*/
    if (cacheIdx == 0)
    {
      partial_sum[get_group_id (0)] = cache[0];
    }
  }
#else
  #error "Double precision floating point not supported."
#endif
