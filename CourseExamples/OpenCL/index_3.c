/* A small program initializing arrays with different values.
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
 * If possible, all four kernels will be started in an
 * out-of-order queue.
 *
 * kernelConstant:	using a constant
 * kernelGroupIdx:	using get_group_id (0)
 * kernelLocalId:	using get_local_id (0)
 * kernelGlobalId:	using get_global_id (0)
 *
 * The OpenCL runtime library doesn't have a function to convert
 * error codes to error names or messages so that the function
 * "getErrorName ()" from file "errorCodes.c" is necessary to
 * convert error codes to error names.
 *
 * The program can be compiled with any C compiler (UNIX: cc, gcc,
 * icc, nvcc, pgcc, ...; Windows: cl, nvcc, ...) if the OpenCL header
 * files and libraries are available.
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
 * index_1.c: determine OpenCL version at runtime
 *		(problem with "nvcc")
 * index_2.c: use "CL_VERSION_2_0", "__NVCC__", and a command
 *		queue with "CL_QUEUE_ON_DEVICE"
 *		(segmentation fault for NVIDIA Quadro K2200)
 * index_3.c: use "CL_VERSION_2_0", "__NVCC__", and a command
 *		queue without "CL_QUEUE_ON_DEVICE"
 *		(segmentation fault for NVIDIA Quadro K2200)
 * index.c:   use "CL_VERSION_2_0", "__NVCC__", and determine
 *		the OpenCL version at runtime
 *		(at last a working version to create a command queue)
 *
 *
 * Compiling:
 *   Linux:
 *     gcc -o index_3 index_3.c errorCodes.c -lOpenCL
 *   Mac OS X:
 *	 Uses frameworks:
 *	   -framework OpenCL
 *	   -framework OpenGL -framework GLUT
 *     gcc -o index_3 index_3.c errorCodes.c -framework OpenCL
 *   Windows:
 *     cl index_3.c errorCodes.c OpenCL.lib
 *
 * Running:
 *   ./index_3
 *
 *
 * File: index_3.c			Author: S. Gross
 * Date: 12.03.2017
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

#define NUM_KERNELS		  4	/* kernels in indexKernel.cl	*/
#define WORK_ITEMS_PER_WORK_GROUP 4
#define WORK_GROUPS_PER_NDRANGE   2
#define VECTOR_SIZE WORK_GROUPS_PER_NDRANGE * WORK_ITEMS_PER_WORK_GROUP

#define FILENAME	 "indexKernel.cl"
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
  cl_int	 ret;			/* OpenCL function return value	*/
  cl_uint	 numPlatforms;		/* # of available devices	*/
  cl_platform_id *platform_ids;		/* platform IDs			*/

  /**************************************************************
   *
   * Step 1: Get all platform IDs
   *
   **************************************************************
   */
  
  /* get the number of available platforms				*/
  numPlatforms = 0;
  ret = clGetPlatformIDs (0, NULL, &numPlatforms);
  if (numPlatforms > 0)
  {
    printf ("\nFound %d platform(s).\n", numPlatforms);
  }
  else
  {
    printf ("\nCouldn't find any OpenCL capable platforms.\n\n");
    return EXIT_SUCCESS;
  }

  /* get platform IDs							*/
  platform_ids = (cl_platform_id *) malloc (numPlatforms *
					    sizeof (cl_platform_id));
  TestEqualsNULL (platform_ids);
  ret = clGetPlatformIDs (numPlatforms, platform_ids, NULL);
  CheckRetValueOfOpenCLFunction (ret);

  /* Try each platform for a GPU device first				*/
  for (unsigned int platform = 0; platform < numPlatforms; ++platform)
  {
    FILE	*fp;			/* for kernelSource		*/
    const char  *kernelSrc;		/* necessary to avoid warning	*/
    char	*kernelSource,
		*kernelDirectory,
		*paramValue,
		*programBuildOptions,
		deviceOpenCL_MajorVersion; /* '1' or '2'		*/
    char	fname[MAX_FNAME_LENGTH];   /* filename incl. pathname	*/
    char *kernelNames[NUM_KERNELS] =
	   {"kernelConstant", "kernelGroupIdx",
	    "kernelLocalId", "kernelGlobalId"
	   };
    char *initText[NUM_KERNELS] =
	   { "Initialization with a constant\n",
	     "\n\nInitialization with group index\n",
	     "\n\nInitialization with local ID \n",
	     "\n\nInitialization with global ID\n"
	   };
    int aVector[NUM_KERNELS][VECTOR_SIZE],	/* array vectors on CPU	*/
	vectorSize,
	retVal,				/* return value			*/
	outOfOrderQueue = 0;		/* 0: no support, 1: support	*/
    size_t	     paramValueSize,
		     kernelSize;	/* size of kernel code		*/
    cl_int	     errcode_ret;	/* OpenCL function return value	*/
    cl_uint	     numDevices;	/* # of available devices	*/
    cl_device_id     device_id;		/* device ID			*/
    cl_context	     context;
    cl_command_queue command_queue;
    cl_mem	     devVector[NUM_KERNELS];	/* vectors on device	*/
    cl_program	     program;
    cl_kernel	     kernel[NUM_KERNELS];
    size_t	     localWorkSize, globalWorkSize;
    cl_context_properties contextProps[3];

    /**************************************************************
     *
     * Step 2: Get a device ID and check if the device supports
     *         OpenCL 2.x and out-of-order queues
     *
     **************************************************************
     */

    printf ("\n********  Using platform %u  ********\n\n", platform);
    /* try to get a device ID for a GPU first and then for a CPU,
     * if a GPU isn't available
     */
    numDevices = 0;
    ret = clGetDeviceIDs (platform_ids[platform], CL_DEVICE_TYPE_GPU,
			  1, &device_id, &numDevices);
    /* Probably returns "CL_DEVICE_NOT_FOUND" so that it is not allowed
     * to use "CheckRetValueOfOpenCLFunction (ret);" The result must be
     * checked with "numDevices".
     */
    if (numDevices == 0)
    {
      printf ("Couldn't find a GPU device on platform %u.\n"
	      "Try to use a CPU next.\n\n", platform);
      ret = clGetDeviceIDs (platform_ids[platform], CL_DEVICE_TYPE_CPU,
			    1, &device_id, &numDevices);
      /* Probably returns "CL_DEVICE_NOT_FOUND" so that it is not
       * allowed to use "CheckRetValueOfOpenCLFunction (ret);" The
       * result must be checked with "numDevices".
       */
      if (numDevices == 0)
      {
	printf ("\nCould neither find a CPU device on platform %u\n."
		"Try next platform.\n\n", platform);
	continue;			/* try next platform		*/
      }
    }

    /* Does the device support OpenCL 2.x?				*/ 
    ret = clGetDeviceInfo (device_id, CL_DEVICE_OPENCL_C_VERSION,
			   0, NULL, &paramValueSize);
    CheckRetValueOfOpenCLFunction (ret);
    paramValue = (char *) malloc (paramValueSize * sizeof (char));
    TestEqualsNULL (paramValue);
    ret = clGetDeviceInfo (device_id, CL_DEVICE_OPENCL_C_VERSION,
			   paramValueSize, paramValue, NULL);
    CheckRetValueOfOpenCLFunction (ret);
    /* paramValue == "OpenCL C x.y ..." where "x" is the major and "y"
     * the minor version number, i.e., paramValue[9] is the major
     * version number (character '1' or '2').
     */
    deviceOpenCL_MajorVersion = paramValue[9];
    free (paramValue);

    /* Does the device support out-of-order queues?			*/ 
    switch (deviceOpenCL_MajorVersion - '0')	/* convert char to int	*/
    {
      case 1:
        {
	  cl_command_queue_properties queueProp = 0;
      
	  ret = clGetDeviceInfo (device_id, CL_DEVICE_QUEUE_PROPERTIES,
				 sizeof (queueProp), &queueProp, NULL);
	  CheckRetValueOfOpenCLFunction (ret);
	  if (queueProp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
	  {
	    outOfOrderQueue = 1;
	  }
	}
	programBuildOptions = NULL;	/* OpenCL 1.2 is default	*/
	break;
      
      case 2:
	outOfOrderQueue     = 1;	/* mandatory for OpenCL 2.x	*/
	programBuildOptions = "-cl-std=CL2.0";
	break;

      default:
	fprintf (stderr, "file: %s  line %d: Unkonown OpenCL major "
		 "version number.\n", __FILE__, __LINE__);
	exit (EXIT_FAILURE);
    }

    /* get device name							*/
    ret = clGetDeviceInfo (device_id, CL_DEVICE_NAME,
			   0, NULL, &paramValueSize);
    CheckRetValueOfOpenCLFunction (ret);
    paramValue =
      (char *) malloc (paramValueSize * sizeof (char));
    TestEqualsNULL (paramValue);
    ret = clGetDeviceInfo (device_id, CL_DEVICE_NAME,
			   paramValueSize, paramValue, NULL);
    CheckRetValueOfOpenCLFunction (ret);
    printf ("Found device %s:\n"
	    "  OpenCL SDK supports OpenCL 2.x:      %s\n"
	    "  Device supports OpenCL 2.x:          %s\n"
	    "  Device supports out-of-order queues: %s\n",
	    paramValue,
	    #if !defined(CL_VERSION_2_0) || defined(__NVCC__)
	      "no",
	    #else
	      "yes",
	    #endif
	    ((deviceOpenCL_MajorVersion - '0') == 1) ? "no" : "yes",
	    (outOfOrderQueue == 0) ? "no" : "yes");
    fflush (stdout);
    free (paramValue);

    /**************************************************************
     *
     * Step 3: Create context for device "device_id"
     *
     **************************************************************
     */

    contextProps[0] = CL_CONTEXT_PLATFORM;
    contextProps[1] = (const cl_context_properties)
			platform_ids[platform];
    contextProps[2] = 0;

    context = clCreateContext (contextProps, 1, &device_id,
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
    /* make sure that the string is \0 terminated			*/
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

    for (int i = 0; i < NUM_KERNELS; ++i)
    {
      devVector[i] = clCreateBuffer (context, CL_MEM_READ_WRITE,
				     VECTOR_SIZE * sizeof (int),
				     NULL, &errcode_ret);
      CheckRetValueOfOpenCLFunction (errcode_ret);
    }

    /**************************************************************
     *
     * Step 8: Create all kernel objects
     *
     **************************************************************
     */

    for (int i = 0; i < NUM_KERNELS; ++i)
    {
      kernel[i] = clCreateKernel (program, kernelNames[i],
				  &errcode_ret);
      CheckRetValueOfOpenCLFunction (errcode_ret);
    }

    /**************************************************************
     *
     * Step 9: Set kernel arguments for all kernels
     *
     **************************************************************
     */

    vectorSize = VECTOR_SIZE;
    for (int i = 0; i < NUM_KERNELS; ++i)
    {
      ret = clSetKernelArg (kernel[i], 0, sizeof (cl_mem),
			    &devVector[i]);
      CheckRetValueOfOpenCLFunction (ret);
      ret = clSetKernelArg (kernel[i], 1, sizeof (int), &vectorSize);
      CheckRetValueOfOpenCLFunction (ret);
    }

    /**************************************************************
     *
     * Step 10: Create command queue for device "device_id"
     *
     **************************************************************
     */

    #if !defined(CL_VERSION_2_0) || defined(__NVCC__)
      if (outOfOrderQueue == 1)
      {
	printf ("  Using out-of-order clCreateCommandQueue (...)\n\n");
	command_queue = clCreateCommandQueue (context, device_id,
			  CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
			  &errcode_ret);
      }
      else
      {
	printf ("  Using in-order clCreateCommandQueue (...)\n\n");
	command_queue = clCreateCommandQueue (context, device_id,
					      0, &errcode_ret);
      }
    #else
      {
	cl_queue_properties queueProps[] =
	  { CL_QUEUE_PROPERTIES,
	    (const cl_queue_properties)
	      (CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE),
	    0
	  };
	printf ("  Using out-of-order "
		"clCreateCommandQueueWithProperties (...)\n\n");
	command_queue = clCreateCommandQueueWithProperties (context,
			  device_id, queueProps, &errcode_ret);
      }
    #endif
    CheckRetValueOfOpenCLFunction (errcode_ret);

    /**************************************************************
     *
     * Step 11: Enqueue a command to execute all kernels
     *
     **************************************************************
     */

    for (int i = 0; i < NUM_KERNELS; ++i)
    {
      localWorkSize  = WORK_ITEMS_PER_WORK_GROUP;
      globalWorkSize = localWorkSize * WORK_GROUPS_PER_NDRANGE;
      ret = clEnqueueNDRangeKernel (command_queue, kernel[i], 1, NULL,
				    &globalWorkSize, &localWorkSize,
				    0, NULL, NULL);
      CheckRetValueOfOpenCLFunction (ret);

      /* finalize							*/
      ret = clFlush (command_queue);
      CheckRetValueOfOpenCLFunction (ret);
      ret = clFinish (command_queue);
      CheckRetValueOfOpenCLFunction (ret);
    }

    /**************************************************************
     *
     * Step 12: Read memory objects (results) from all kernels
     *
     **************************************************************
     */

    for (int i = 0; i < NUM_KERNELS; ++i)
    {
      ret = clEnqueueReadBuffer (command_queue, devVector[i], CL_TRUE,
				 0, VECTOR_SIZE * sizeof (int),
				 &aVector[i], 0, NULL, NULL);
      CheckRetValueOfOpenCLFunction (ret);
    }

    /* print results							*/
    for (int kern = 0; kern < NUM_KERNELS; ++kern)
    {
      printf ("%s", initText[kern]);
      for (int i = 0; i < VECTOR_SIZE; ++i)
      {
	printf ("  %d", aVector[kern][i]);
      }
    }
    printf ("\n\n");

    /**************************************************************
     *
     * Step 13: Free objects
     *
     **************************************************************
     */

    for (int i = 0; i < NUM_KERNELS; ++i)
    {
      ret = clReleaseKernel (kernel[i]);
      CheckRetValueOfOpenCLFunction (ret);
    }
    ret = clReleaseProgram (program);
    CheckRetValueOfOpenCLFunction (ret);
    for (int i = 0; i < NUM_KERNELS; ++i)
    {
      ret = clReleaseMemObject (devVector[i]);
      CheckRetValueOfOpenCLFunction (ret);
    }
    ret = clReleaseCommandQueue (command_queue);
    CheckRetValueOfOpenCLFunction (ret);
    ret = clReleaseContext (context);
    CheckRetValueOfOpenCLFunction (ret);
  }

  return EXIT_SUCCESS;
}
