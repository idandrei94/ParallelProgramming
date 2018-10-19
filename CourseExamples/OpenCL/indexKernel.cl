/* Four kernels that initialize an array with different values.
 *
 *
 * File: index.cl			Author: S. Gross
 * Date: 29.12.2016
 *
 */


/* Initialize vector "a" with different values using GPU threads.
 *
 * Input:		vecSize		vector size
 * Output		a		initialized array
 * Return value:	none
 * Sideeffects:		none
 *
 */
__kernel void kernelConstant (__global int * restrict a,
			      const int vecSize)
{
  int idx = get_global_id (0);

  if (idx < vecSize)
  {
    a[idx] = 9;
  }
}


__kernel void kernelGroupIdx (__global int * restrict a,
			      const int vecSize)
{
  int idx = get_global_id (0);

  if (idx < vecSize)
  {
    a[idx] = get_group_id (0);
  }
}


__kernel void kernelLocalId (__global int * restrict a,
			     const int vecSize)
{
  int idx = get_global_id (0);

  if (idx < vecSize)
  {
    a[idx] = get_local_id (0);
  }
}


__kernel void kernelGlobalId (__global int * restrict a,
			      const int vecSize)
{
  int idx = get_global_id (0);

  if (idx < vecSize)
  {
    a[idx] = idx;
  }
}
