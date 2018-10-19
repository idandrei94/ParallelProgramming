/* Some constants for "dot_prod_OpenCL.c" and "dotProdKerenl.cl".
 *
 *
 * File: dot_prod_OpenCL.h		Author: S. Gross
 * Date: 14.10.2016
 *
 */

/* Not enough memory on HD Graphics 4600 if VECTOR_SIZE > 53.398.732	*/
#define	VECTOR_SIZE		  10000000	/* vector size (10^7)	*/
#define WORK_ITEMS_PER_WORK_GROUP 128		/* power of two	required*/
#define WORK_GROUPS_PER_NDRANGE    32

/* "barrier ()" has been renamed as "work_group_barrier ()" in
 * OpenCL 2.0
 */
#ifdef CL_VERSION_2_0
  #define barrier work_group_barrier
#endif
