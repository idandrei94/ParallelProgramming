/* OpenCL GPU program with a kernel printing "Hello ...".
 *
 * This version reads the "kernel" from a file. The environment
 * variable KERNEL_FILES must point to the directory of the kernel
 * file, if you want to run the program from any directory.
 * Otherwise the program can only be executed successfully, if the
 * kernel file is available in the current directory. The pathname
 * must start with a drive letter, if you use Cygwin or Windows and
 * and if the program and kernel files may be located on different
 * drives, e.g.,
 * Cygwin:  "setenv KERNEL_FILES c:/cygwin64/${HOME}/kernel_files",
 * Windows: "set KERNEL_FILES=c:\temp\kernel_files".
 *
 * The OpenCL runtime library doesn't have a function to convert
 * error codes to error names or messages so that the function
 * "getErrorName ()" from file "errorCodes.c" is necessary to
 * convert error codes to error names.
 *
 * The program can be compiled with any C compiler (UNIX: cc, gcc,
 * icc, nvcc, pgcc, ...; Windows: cl, nvcc, ...) if the OpenCL
 * header files and libraries are available.
 *
 * "clCreateCommandQueue ()" is deprecated since OpenCl 2.0, but
 * some OpenCL implementations only support OpenCL 1.2, so that
 * the new function "clCreateCommandQueueWithProperties ()" isn't
 * available. Some OpenCL 1.2 implementations support out-of-order
 * queues and some don't, so that you must verify that a special
 * feature is available, before you use it (otherwise a function
 * returns an error or the program even breaks when you execute it).
 * If you check the OpenCL version at runtime, you get at least a
 * warning about the deprecated "clCreateCommandQueue ()" function,
 * when you compile the program. If you choose the correct function
 * version only with "#ifndef CL_VERSION_2_0" at compile time, you
 * may run into a segmentation fault, when you execute the program,
 * if the device only supports OpenCL 1.2. This happens for example
 * for the NVIDIA Quadro K2200 graphics card if you use Intel or
 * AMD OpenCL on SuSE Linux Enterprise 12.1. Therefore, it is
 * necessary to combine "CL_VERSION_2_0" and a check at runtime to
 * be able to compile the program with OpenCL from AMD, Intel, and
 * NVIDIA with all compilers and to get a working executable.
 * Unfortunately you must live with a warning about a deprecated
 * function in that case.
 *
 * NVIDIA "nvcc" uses "/usr/local/cuda/bin/nvcc.profile" to set
 * environment variables for its own header and library files.
 * Therefore, the header and library files from the OpenCL
 * implementation which you may have selected in $HOME/.cshrc
 * will not be used at compile time. At run-time the libraries
 * from the selected OpenCL distribution will be used. At the
 * moment NVIDIA supports only OpenCL 1.2 (CUDA-7.5 and CUDA-8.0),
 * so that things are even more complicated. Fortunately NVIDIA
 * defines "__NVCC__", so that you can select the correct function
 * for "nvcc" as well.
 *
 * Creating a command queue is very complicated at the moment, if
 * you want to use the functions "clCreateCommandQueue()" and
 * "clCreateCommandQueueWithProperties()" in a heterogeneous OpenCL
 * 1.2 and 2.x environment for different compilers. You can sometimes
 * suppress the warning about a deprecated function, if you add
 * "#define CL_USE_DEPRECATED_OPENCL_1_2_APIS" before "#include CL.h".
 *
 * Therefore, only "index*.c" try to use the new function
 * "clCreateCommandQueueWithProperties()" if possible. All other
 * OpenCL files use the old function "clCreateCommandQueue()". All
 * OpenCL files try to suppress the warning about a deprecated
 * OpenCL 1.2 function.
 *
 *
 * Compiling:
 *   Linux:
 *     gcc -o hello_OpenCL hello_OpenCL.c errorCodes.c -lOpenCL
 *   Mac OS X:
 *	 Uses frameworks:
 *	   -framework OpenCL
 *	   -framework OpenGL -framework GLUT
 *     gcc -o hello_OpenCL hello_OpenCL.c errorCodes.c \
 *	   -framework OpenCL
 *   Windows:
 *     cl hello_OpenCL.c errorCodes.c OpenCL.lib
 *
 * Running:
 *   ./hello_OpenCL
 *
 *
 * File: hello_OpenCL.c			Author: S. Gross
 * Date: 12.06.2017
 *
 */

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __APPLE__
  #include <CL/cl.h>
#else
  #include <OpenCL/opencl.h>
#endif

#define WORK_ITEMS_PER_WORK_GROUP 3
#define WORK_GROUPS_PER_NDRANGE   2

#define FILENAME	 "helloKernel.cl"
#define KERNEL_NAME	 "helloKernel"
#define MAX_FNAME_LENGTH 256

#if (defined(WIN32) || defined(_WIN32) || defined(Win32)) && \
     !defined(Cygwin)
  #define PATH_SEPARATOR "\\" 
#else 
  #define PATH_SEPARATOR "/" 
#endif 


/* define macro to test the result of a "malloc" operation		*/
#define TestEqualsNULL(val)  \
  if (val == NULL) \
  { \
    fprintf (stderr, "file: %s  line %d: Couldn't allocate memory.\n", \
	     __FILE__, __LINE__); \
    exit (EXIT_FAILURE); \
  }


/* Define macro to check the return value of an OpenCL function. The
 * function prototype is necessary, because the compiler will assume
 * that "getErrorName ()" will return "int" without a prototype.
 */
const char *getErrorName (cl_int errCode);
#define CheckRetValueOfOpenCLFunction(val) \
  if (val != CL_SUCCESS) \
  { \
    fprintf (stderr, "file: %s  line %d: %s.\n", \
	     __FILE__, __LINE__, getErrorName (val)); \
    exit (EXIT_FAILURE); \
  }



int main (void)
{
  FILE		 *fp;			/* for kernelSource		*/
  const char	 *kernelSrc;		/* necessary to avoid warning	*/
  char		 *kernelSource,
		 *kernelDirectory,
		 *paramValue,
		 *platformName,
		 *deviceName,
		 *programBuildOptions = NULL;
  char		 fname[MAX_FNAME_LENGTH]; /* filename incl. pathname	*/
  int		 retVal;		/* return value			*/
  size_t	 kernelSize,		/* size of kernel code		*/
		 paramValueSize;
  cl_int	 errcode_ret,		/* returned error code		*/
 		 ret;			/* OpenCL function return value	*/
  cl_uint	 numDevices;		/* # of available devices	*/
  cl_platform_id platform_id = NULL;	/* platform ID			*/
  cl_device_id	 device_id;		/* device ID			*/
  cl_context	 context;
  cl_command_queue command_queue;
  cl_program	   program;
  cl_kernel	   kernel;
  size_t	   localWorkSize, globalWorkSize;

  /**************************************************************
   *
   * Step 1: Get a platform ID
   *
   **************************************************************
   */
  
  /* get platform ID of the first platform				*/
  ret = clGetPlatformIDs (1, &platform_id, NULL);
  CheckRetValueOfOpenCLFunction (ret);

  /* get platform name							*/
  ret = clGetPlatformInfo (platform_id, CL_PLATFORM_NAME,
			   0, NULL, &paramValueSize);
  CheckRetValueOfOpenCLFunction (ret);
  platformName = (char *) malloc (paramValueSize * sizeof (char));
  TestEqualsNULL (platformName);
  ret = clGetPlatformInfo (platform_id, CL_PLATFORM_NAME,
			   paramValueSize, platformName, NULL);
  CheckRetValueOfOpenCLFunction (ret);

  /**************************************************************
   *
   * Step 2: Get a device ID
   *
   **************************************************************
   */
  
  /* get device ID for the first device					*/
  numDevices = 0;
  ret = clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_ALL,
			1, &device_id, &numDevices);
  if (numDevices > 0)
  {
    /* get device name							*/
    ret = clGetDeviceInfo (device_id, CL_DEVICE_NAME,
			   0, NULL, &paramValueSize);
    CheckRetValueOfOpenCLFunction (ret);
    deviceName = (char *) malloc (paramValueSize * sizeof (char));
    TestEqualsNULL (deviceName);
    ret = clGetDeviceInfo (device_id, CL_DEVICE_NAME,
			   paramValueSize, deviceName, NULL);
    CheckRetValueOfOpenCLFunction (ret);
    printf ("\nFound device \"%s\"\n"
	    "  on platform \"%s\".\n", deviceName, platformName);
    free (deviceName);
  }
  else
  {
    printf ("\nCouldn't find any devices on platform \"%s\".\n",
	    platformName);
  }
  free (platformName);

  /**************************************************************
   *
   * Step 3: Create context for device "device_id"
   *
   **************************************************************
   */

  context = clCreateContext (NULL, 1, &device_id,
			     NULL, NULL, &errcode_ret);
  CheckRetValueOfOpenCLFunction (errcode_ret);

  /**************************************************************
   *
   * Step 4: Create kernel as a string
   *
   **************************************************************
   */

  /* read kernel from file						*/
  memset (fname, 0, MAX_FNAME_LENGTH);
  kernelDirectory = getenv ("KERNEL_FILES");
  if (kernelDirectory != NULL)
  {
    strncpy (fname, kernelDirectory, MAX_FNAME_LENGTH - 1);
    /* check, if the last character of the environment variable
     * KERNEL_FILES is a path separator and add one otherwise
     */
    if (fname[strlen (fname) - 1] != PATH_SEPARATOR[0])
    {
      strncat (fname, PATH_SEPARATOR,
	       MAX_FNAME_LENGTH - strlen (fname) - 1);
    }
  }
  strncat (fname, FILENAME, MAX_FNAME_LENGTH - strlen (fname) - 1);
  fp = fopen (fname, "r");
  if (fp == NULL)
  {
    fprintf (stderr, "file: %s  line %d: Couldn't open file "
	     "\"%s\".\n", __FILE__, __LINE__, fname);
    exit (EXIT_FAILURE);
  }
  retVal = fseek (fp, 0, SEEK_END);
  if (retVal != 0)
  {
    fprintf (stderr, "file: %s  line %d: \"fseek ()\" failed: "
	     "\"%s\".\n", __FILE__, __LINE__, strerror (retVal));
    exit (EXIT_FAILURE);
  }
  kernelSize = (size_t) ftell (fp);
  rewind (fp);
  kernelSource = (char *) malloc (kernelSize + 1);
  TestEqualsNULL (kernelSource);
  /* make sure that the string is \0 terminated				*/
  kernelSource[kernelSize] = '\0';
  fread (kernelSource, sizeof (char), kernelSize, fp);
  if (ferror (fp) != 0)
  {
    fprintf (stderr, "file: %s  line %d: \"fread ()\" failed.\n",
	     __FILE__, __LINE__);
    exit (EXIT_FAILURE);
  }
  fclose (fp);
 
  /**************************************************************
   *
   * Step 5: Create program object
   *
   **************************************************************
   */

  /* Without "kernelSrc" "gcc -Wcast-qual" displays the following
   * warning "... warning: to be safe all intermediate pointers in
   * cast from 'char **' to 'const char **' must be 'const'
   * qualified [-Wcast-qual]". 
   */
  kernelSrc = kernelSource;
  program = clCreateProgramWithSource (context, 1,
	      (const char **) &kernelSrc, NULL, &errcode_ret);
  CheckRetValueOfOpenCLFunction (errcode_ret);
  kernelSrc = NULL;
  free (kernelSource);

  /**************************************************************
   *
   * Step 6: Build program executables
   *
   **************************************************************
   */

  ret = clBuildProgram (program, 1, &device_id,
			programBuildOptions, NULL, NULL);
  if (ret != CL_SUCCESS)
  {
    /* check log file							*/
    ret = clGetProgramBuildInfo (program, device_id,
				 CL_PROGRAM_BUILD_LOG, 0, NULL,
				 &paramValueSize);
    CheckRetValueOfOpenCLFunction (ret);
    if (paramValueSize > 0)
    {
      paramValue =
	(char *) malloc ((paramValueSize * sizeof (char)) + 1);
      TestEqualsNULL (paramValue);
      /* make sure that the string is \0 terminated			*/
      paramValue[paramValueSize] = '\0';
      ret = clGetProgramBuildInfo (program, device_id,
				   CL_PROGRAM_BUILD_LOG,
				   paramValueSize, paramValue, NULL);
      CheckRetValueOfOpenCLFunction (ret);
      printf ("\nCompiler log file:\n\n%s", paramValue);
      free (paramValue);
      exit (EXIT_FAILURE);
    }
  }

  /**************************************************************
   *
   * Step 7: Create buffer for all kernels on device "device_id"
   *
   **************************************************************
   */

  /* nothing to do, because the "kernel" doesn't have any parameters	*/

  /**************************************************************
   *
   * Step 8: Create kernel object
   *
   **************************************************************
   */

  kernel = clCreateKernel (program, KERNEL_NAME, &errcode_ret);
    CheckRetValueOfOpenCLFunction (errcode_ret);

  /**************************************************************
   *
   * Step 9: Set kernel arguments for all kernels
   *
   **************************************************************
   */

  /* nothing to do, because the "kernel" doesn't have any parameters	*/

  /**************************************************************
   *
   * Step 10: Create command queue for device "device_id"
   *
   **************************************************************
   */

  command_queue = clCreateCommandQueue (context, device_id, 0,
					&errcode_ret);
  CheckRetValueOfOpenCLFunction (errcode_ret);

  /**************************************************************
   *
   * Step 11: Enqueue a command to execute the kernel
   *
   **************************************************************
   */

    localWorkSize  = WORK_ITEMS_PER_WORK_GROUP;
    globalWorkSize = localWorkSize * WORK_GROUPS_PER_NDRANGE;
    ret = clEnqueueNDRangeKernel (command_queue, kernel, 1, NULL,
				  &globalWorkSize, &localWorkSize,
				  0, NULL, NULL);
    CheckRetValueOfOpenCLFunction (ret);

    /* finalize								*/
    ret = clFlush (command_queue);
    CheckRetValueOfOpenCLFunction (ret);
    ret = clFinish (command_queue);
    CheckRetValueOfOpenCLFunction (ret);

  /**************************************************************
   *
   * Step 12: Read memory objects (results) from kernel
   *
   **************************************************************
   */

  /* nothing to do, because the "kernel" doesn't have any parameters	*/

  /**************************************************************
   *
   * Step 13: Free objects
   *
   **************************************************************
   */

  ret = clReleaseKernel (kernel);
  CheckRetValueOfOpenCLFunction (ret);
  ret = clReleaseProgram (program);
  CheckRetValueOfOpenCLFunction (ret);
  ret = clReleaseCommandQueue (command_queue);
  CheckRetValueOfOpenCLFunction (ret);
  ret = clReleaseContext (context);
  CheckRetValueOfOpenCLFunction (ret);

  return EXIT_SUCCESS;
}
