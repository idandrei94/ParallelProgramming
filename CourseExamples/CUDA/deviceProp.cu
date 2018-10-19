/* Print some device properties.
 *
 *
 * Compiling:
 *   nvcc -arch=sm_50 -o deviceProp deviceProb.cu
 *   clang --cuda-gpu-arch=sm_50 -o deviceProp deviceProb.cu -lcudart
 *
 * Running:
 *   ./deviceProp
 *
 *
 * File: deviceProp.cu			Author: S. Gross
 * Date: 14.02.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>


int getNumCores (int major, int minor);


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
  int numDevices;			/* number of available devices	*/
  cudaDeviceProp prop;			/* device properties		*/
  cudaError_t	 ret;			/* CUDA function return value	*/

  ret = cudaGetDeviceCount (&numDevices);
  CheckRetValueOfCudaFunction (ret);
  printf ("\nFound %d CUDA capable device(s).\n", numDevices);
  for (int i = 0; i < numDevices; ++i)
  {
    ret = cudaGetDeviceProperties (&prop, i);
    CheckRetValueOfCudaFunction (ret);
    printf ("\nSome properties of device %d\n", i);
    /* "totalGlobalMem", "totalConstMem", and "sharedMemPerBlock"
     * have type "size_t" so that "%zu" would be the favoured format
     * specifier in a printf-statement. Unfortunately, Microsoft
     * Visual Studio doesn't support "%zu" with older versions,
     * so that "%lu" is a better and portable choice.
     */
    printf ("  Name:                                         %s\n"
	    "  Compute capability:                           %d.%d\n"
	    "  Clock rate:                                   %6.2f MHz\n"
	    "  Total global memory:                          %lu GB\n"
	    "  Total constant memory:                        %lu KB\n"
	    "  L2 cache size:                                %d KB\n"
	    "  Can map host memory into device memory:       %s\n"
	    "  Unified address space with the host:          %s\n"
	    "  Overlapping device copy and kernel execution: %s\n"
	    "  Supports concurrent kernels:                  %s\n"
	    "  Number of multiprocessors:                    %d\n"
	    "  Number of CUDA cores:                         %d\n"
	    "  Number of CUDA cores per multiprocessor:      %d\n"
	    "  Threads per warp:                             %d\n"
	    "  Max. threads per multiprocessor:              %d\n"
	    "  Max. warps per multiprocessor:                %d\n"
	    "  Max. threads per block:                       %d\n"
	    "  Shared memory per block:                      %lu KB\n"
	    "  Registers per multiprocessor:                 %d\n"
	    "  Registers per block:                          %d\n"
	    "  Max. thread dimensions:                   "
	    "    (%d, %d, %d)\n"
	    "  Max. grid dimensions:                     "
	    "    (%d, %d, %d)\n",
	    prop.name,
	    prop.major, prop.minor,
	    (double) prop.clockRate / 1000,
	    (long unsigned int) prop.totalGlobalMem / (1024 * 1024 * 1024),
	    (long unsigned int) prop.totalConstMem / 1024,
	    prop.l2CacheSize / 1024,
	    (prop.canMapHostMemory) ? "Yes" : "No",
	    (prop.unifiedAddressing) ? "Yes" : "No",
	    (prop.asyncEngineCount == 0) ? "Not possible" : \
	      (prop.asyncEngineCount == 1) ? \
	        "Yes, either upload or download" : \
		"Yes, concurrent upload and download",
	    (prop.concurrentKernels) ? "Yes" : "No",
	    prop.multiProcessorCount,
            getNumCores (prop.major, prop.minor) *
	      prop.multiProcessorCount,
            getNumCores (prop.major, prop.minor),
	    prop.warpSize,
	    prop.maxThreadsPerMultiProcessor,
	    prop.maxThreadsPerMultiProcessor / prop.warpSize,
	    prop.maxThreadsPerBlock,
	    (long unsigned int) prop.sharedMemPerBlock / 1024,
	    prop.regsPerMultiprocessor,
	    prop.regsPerBlock,
	    prop.maxThreadsDim[0], prop.maxThreadsDim[1],
	    prop.maxThreadsDim[2],
	    prop.maxGridSize[0], prop.maxGridSize[1],
	    prop.maxGridSize[2]);
  }

  /* reset current device						*/
  ret = cudaDeviceReset ();
  CheckRetValueOfCudaFunction (ret);

  return EXIT_SUCCESS;
}


/* The following function is based on and corresponds to the function
 * "_ConvertSMVer2Cores()" from the file
 * "cuda-7.5/samples/common/inc/helper_cuda.h" of the "CUDA Toolkit
 * 7.5". It uses the compute capability to determine the number of
 * cores per streaming multiprocessor.
 *
 */
int getNumCores (int major, int minor)
{
  /* Defines for the GPU Architecture types using the SM (streaming
   * multiprocessor) version to determine the number of cores per SM.
   *
   * 0xMm (hexidecimal notation)
   *   M = SM Major version
   *   m = SM minor version
   */
  typedef struct
  {
    int SM;
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] =
  {
    {0x20, 32 },	/* Fermi Generation (SM 2.0) GF100 class	*/
    {0x21, 48 },	/* Fermi Generation (SM 2.1) GF10x class	*/
    {0x30, 192},	/* Kepler Generation (SM 3.0) GK10x class	*/
    {0x32, 192},	/* Kepler Generation (SM 3.2) GK10x class	*/
    {0x35, 192},	/* Kepler Generation (SM 3.5) GK11x class	*/
    {0x37, 192},	/* Kepler Generation (SM 3.7) GK21x class	*/
    {0x50, 128},	/* Maxwell Generation (SM 5.0) GM10x class	*/
    {0x52, 128},	/* Maxwell Generation (SM 5.2) GM20x class	*/
    {  -1, -1 }
  };

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1)
  {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
    {
      return nGpuArchCoresPerSM[index].Cores;
    }
    index++;
  }

  /* If we don't find the values, we default use the previous one to
   * run properly.
   */
  printf ("MapSMtoCores for SM %d.%d is undefined. Default to use "
	  "%d Cores/SM\n",
	  major, minor, nGpuArchCoresPerSM[index-1].Cores);
  return nGpuArchCoresPerSM[index-1].Cores;
}
