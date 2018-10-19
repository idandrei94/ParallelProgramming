/* "Kernel" for OpenCL program printing "Hello ...".
 *
 *
 * File: helloKernel.cl			Author: S. Gross
 * Date: 19.05.2017
 *
 */

__kernel void helloKernel (void)
{
  /* "get_local_id ()" and "get_group_id ()" return a value of
   * type "size_t" so that "%zu" would be the favoured format
   * specifier in a printf-statement. Unfortunately, Microsoft
   * Visual Studio doesn't support "%zu" with older versions,
   * so that "%llu" is a better and portable choice.
   */
  printf ("Hello from work-item %llu in work-group %llu.\n",
  	  (long long unsigned int) get_local_id (0),
	  (long long unsigned int) get_group_id (0));
}
