/* GPU program with a kernel printing "Hello ...". The program
 * checks if the kernel launch was successful or not and synchronizes
 * its output with the kernel.
 *
 *
 * Compiling:
 *   nvcc -arch=sm_50 -o hello_kernel_3 hello_kernel_3.cu
 *   clang --cuda-gpu-arch=sm_50 -o hello_kernel_3 \
 *	   hello_kernel_3.cu -lcudart
 *
 * Running:
 *   ./hello_kernel_3
 *
 *
 * File: hello_kernel_3.cu		Author: S. Gross
 * Date: 14.02.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>

__global__ void helloKernel (void);

/* define macro to check the return value of a CUDA function		*/
#define CheckRetValueOfCudaFunction(val) \
  if (val != cudaSuccess) \
  { \
    fprintf (stderr, "file: %s  line %d: %s.\n", \
	     __FILE__, __LINE__, cudaGetErrorString (val)); \
    cudaDeviceReset (); \
    exit (EXIT_FAILURE); \
  }
    

int main (void)
{
  cudaError_t ret;			/* CUDA function return value	*/

  printf ("Call my \"hello\" kernel.\n");
  helloKernel <<<2, 3>>> ();
  printf ("Have called my \"hello\" kernel and wait for its "
	  "completion.\n");
  ret = cudaDeviceSynchronize ();
  CheckRetValueOfCudaFunction (ret);
  printf ("I terminate now.\n");

  /* reset current device						*/
  ret = cudaDeviceReset ();
  CheckRetValueOfCudaFunction (ret);

  return EXIT_SUCCESS;
}


__global__ void helloKernel (void)
{
  printf ("Hello from thread %d in thread block %d.\n",
	  threadIdx.x, blockIdx.x);
}
