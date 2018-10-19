/* GPU program with a kernel printing "Hello ...". The program
 * doesn't check if the kernel launch was successful or not.
 *
 *
 * Compiling:
 *   nvcc -arch=sm_50 -o hello_kernel_1 hello_kernel_1.cu
 *   clang --cuda-gpu-arch=sm_50 -o hello_kernel_1 \
 *	   hello_kernel_1.cu -lcudart
 *
 * Running:
 *   ./hello_kernel_1
 *
 *
 * File: hello_kernel_1.cu		Author: S. Gross
 * Date: 14.02.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>

__global__ void helloKernel (void);

int main (void)
{
  printf ("Call my \"hello\" kernel.\n");
  helloKernel <<<2, 3>>> ();
  printf ("Have called my \"hello\" kernel and terminate now.\n");

  /* reset current device						*/
  cudaDeviceReset ();
  return EXIT_SUCCESS;
}


__global__ void helloKernel (void)
{
  printf ("Hello from thread %d in thread block %d.\n",
	  threadIdx.x, blockIdx.x);
}
