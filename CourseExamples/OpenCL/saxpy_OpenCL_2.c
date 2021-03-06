/* Simplified implementation of the SAXPY subprogram (single
 * precision alpha x plus y) from the Basic Linear Algebra
 * Subprogram library (BLAS).
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
 *     gcc -o saxpy_OpenCL_2 saxpy_OpenCL_2.c errorCodes.c -lOpenCL
 *   Mac OS X:
 *	 Uses frameworks:
 *	   -framework OpenCL
 *	   -framework OpenGL -framework GLUT
 *     gcc -o saxpy_OpenCL_2 saxpy_OpenCL_2.c errorCodes.c \
 *	   -framework OpenCL
 *   Windows:
 *     cl saxpy_OpenCL_2.c errorCodes.c OpenCL.lib
 *
 * Running:
 *   ./saxpy_OpenCL_2
 *
 *
 * File: saxpy_OpenCL_2.c		Author: S. Gross
 * Date: 12.06.2017
 *
 */

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <assert.h>
#ifndef __APPLE__
  #include <CL/cl.h>
#else
  #include <OpenCL/opencl.h>
#endif

/* Possibly wrong results if VECTOR_SIZE > 3.355.444 due to rounding
 * errors for float values. Initializing vectors in step 5 results in
 * the following values.
 *
 * VECTOR_SIZE = 3355444   y[0] =   16777220.00
 * VECTOR_SIZE = 3355444   y[1] =   16777215.00
 * VECTOR_SIZE = 3355444   y[2] =   16777210.00
 * VECTOR_SIZE = 3355444   y[3] =   16777205.00
 *
 * VECTOR_SIZE = 3355445   y[0] =   16777224.00  must be: 16777225.00
 * VECTOR_SIZE = 3355445   y[1] =   16777220.00
 * VECTOR_SIZE = 3355445   y[2] =   16777215.00
 * VECTOR_SIZE = 3355445   y[3] =   16777210.00
 *
 * VECTOR_SIZE should be lower or equal to (2^FLT_MANT_DIG / ALPHA)
 * to avoid rounding errors.
 * 
 */
#define	VECTOR_SIZE 1000000		/* vector size (10^6)		*/
#define ALPHA	    5.0F		/* scalar alpha			*/
#define EPS	    FLT_EPSILON		/* from float.h (1.19...e-07)	*/

#define KERNEL_NAME "saxpyKernel"
#define FILENAME    "saxpyKernel.cl"
#define MAX_FNAME_LENGTH	  256
#define WORK_ITEMS_PER_WORK_GROUP 128
#define WORK_GROUPS_PER_NDRANGE    32

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


/* heap memory to avoid a segmentation fault due to a stack overflow	*/
static float x[VECTOR_SIZE],
	     y[VECTOR_SIZE];



int main (void)
{
  cl_int	 ret;			/* OpenCL function return value	*/
  cl_uint	 numPlatforms;		/* # of available platforms	*/
  cl_platform_id *platform_ids;		/* list of platforms		*/

  assert (VECTOR_SIZE <= exp2f (FLT_MANT_DIG) / ALPHA);

  /**************************************************************
   *
   * Step 1: Get a list of available platforms
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

  /* execute the kernel on all available devices on all platforms	*/
  for (unsigned int platform = 0; platform < numPlatforms; ++platform)
  {
    char	 *platformName, **deviceName;
    size_t	 paramValueSize;
    cl_int	 errcode_ret;		/* returned error code		*/
    cl_uint	 numDevices;		/* # of available devices	*/
    cl_device_id *device_ids;

    /* get platform name						*/
    ret = clGetPlatformInfo (platform_ids[platform], CL_PLATFORM_NAME,
			     0, NULL, &paramValueSize);
    CheckRetValueOfOpenCLFunction (ret);
    platformName = (char *) malloc (paramValueSize * sizeof (char));
    TestEqualsNULL (platformName);
    ret = clGetPlatformInfo (platform_ids[platform], CL_PLATFORM_NAME,
			     paramValueSize, platformName, NULL);
    CheckRetValueOfOpenCLFunction (ret);

    /**************************************************************
     *
     * Step 2: Get a list of available devices
     *
     **************************************************************
     */
  
    /* get the number of available devices				*/
    numDevices = 0;
    ret = clGetDeviceIDs (platform_ids[platform], CL_DEVICE_TYPE_ALL,
			  0, NULL, &numDevices);
    CheckRetValueOfOpenCLFunction (ret);
    if (numDevices > 0)
    {
      printf ("\n\n\nFound the following %d device(s) on platform "
	      "\"%s\".\n", numDevices, platformName);
    }
    else
    {
      printf ("\nCouldn't find any devices on platform \"%s\".\n",
	      platformName);
      break;				/* try next platform		*/
    }
    free (platformName);


    /* get device IDs							*/
    device_ids = (cl_device_id *) malloc (numDevices *
					  sizeof (cl_device_id));
    TestEqualsNULL (device_ids);
    ret = clGetDeviceIDs (platform_ids[platform], CL_DEVICE_TYPE_ALL,
			  numDevices, device_ids, NULL);
    CheckRetValueOfOpenCLFunction (ret);

    /* get device names							*/
    deviceName = (char **) malloc (numDevices * sizeof (char *));
    TestEqualsNULL (deviceName);
    for (unsigned int device = 0; device < numDevices; ++device)
    {
      ret = clGetDeviceInfo (device_ids[device], CL_DEVICE_NAME,
			     0, NULL, &paramValueSize);
      CheckRetValueOfOpenCLFunction (ret);
      deviceName[device] =
	(char *) malloc (paramValueSize * sizeof (char));
      TestEqualsNULL (deviceName[device]);
      ret = clGetDeviceInfo (device_ids[device], CL_DEVICE_NAME,
			     paramValueSize, deviceName[device], NULL);
      CheckRetValueOfOpenCLFunction (ret);
      printf ("  %s\n", deviceName[device]);
    }

    for (unsigned int device = 0; device < numDevices; ++device)
    {
      FILE	  *fp;			/* for kernelSource		*/
      const char  *kernelSrc;		/* necessary to avoid warning	*/
      char	  *kernelSource,
		  *kernelDirectory;
      size_t	  kernelSize;		/* size of kernel code		*/
      char	  fname[MAX_FNAME_LENGTH]; /* filename incl. pathname	*/
      float       tmp_y0,		/* avoid code removing		*/
		  alpha;
      int	  tmp_diff,		/* result ok?			*/
		  retVal,		/* return value			*/
		  vectorSize;
      time_t      start_wall, end_wall;		/* start/end time (CPU)	*/
      clock_t     cpu_time;			/* used cpu time	*/
      cl_ulong    clStartTime, clEndTime;	/* start/end time (GPU)	*/
      cl_context	context;
      cl_command_queue	command_queue;
      cl_event		profileEvent;
      cl_mem		x_dev, y_dev;
      cl_program	program;
      cl_kernel		kernel;
      char		*paramValue,
			*programBuildOptions = NULL;
      size_t		localWorkSize, globalWorkSize;
      cl_context_properties contextProps[] =
	{ CL_CONTEXT_PLATFORM,
	  (const cl_context_properties) platform_ids[platform],
	  0
	};
      cl_command_queue_properties queueProps =
	CL_QUEUE_PROFILING_ENABLE;

      printf ("\n\nUsing device \"%s\".\n\n", deviceName[device]);
      free (deviceName[device]);

      /**************************************************************
       *
       * Step 3: Create context for device device_ids[device]
       *
       **************************************************************
       */

      context = clCreateContext (contextProps, 1, &device_ids[device],
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
      kernelSrc = kernelSource;
      program = clCreateProgramWithSource (context, 1,
		  (const char **) &kernelSrc, NULL, &errcode_ret);
      CheckRetValueOfOpenCLFunction (errcode_ret);
      free (kernelSource);

      /**************************************************************
       *
       * Step 6: Build program executables
       *
       **************************************************************
       */

      ret = clBuildProgram (program, 1, &device_ids[device],
			    programBuildOptions, NULL, NULL);
      if (ret != CL_SUCCESS)
      {
	/* check log file						*/
	ret = clGetProgramBuildInfo (program, device_ids[device],
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
	  ret = clGetProgramBuildInfo (program, device_ids[device],
				       CL_PROGRAM_BUILD_LOG,
				       paramValueSize, paramValue, NULL);
	  CheckRetValueOfOpenCLFunction (ret);
	  printf ("\nCompiler log file:\n\n%s", paramValue);
	  free (paramValue);
	  continue;
	}
      }

      /**************************************************************
       *
       * Step 7: Create buffers for device device_ids[device]
       *
       **************************************************************
       */

      /* Initialize both vectors. The saxpy function computes
       * y = alpha * x + y. With the following initialization we get
       * constant values for the resulting vector.
       * new_y[i] = alpha * x[i] + y[i]
       *	      = alpha * i + alpha * (VECTOR_SIZE - i)
       *	      = alpha * VECTOR_SIZE
       */
      for (int i = 0; i < VECTOR_SIZE; ++i)
      {
	x[i] = (float) i;
	y[i] = ALPHA * (float) (VECTOR_SIZE - i);
      }
      x_dev = clCreateBuffer (context,
			      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			      VECTOR_SIZE * sizeof (float), x,
			      &errcode_ret);
      CheckRetValueOfOpenCLFunction (errcode_ret);
      y_dev = clCreateBuffer (context, CL_MEM_COPY_HOST_PTR,
			      VECTOR_SIZE * sizeof (float), y,
			      &errcode_ret);
      CheckRetValueOfOpenCLFunction (errcode_ret);

      /**************************************************************
       *
       * Step 8: Create kernel objects
       *
       **************************************************************
       */

      kernel = clCreateKernel (program, KERNEL_NAME, &errcode_ret);
      CheckRetValueOfOpenCLFunction (errcode_ret);

      /**************************************************************
       *
       * Step 9: Set kernel arguments
       *
       **************************************************************
       */

      vectorSize = VECTOR_SIZE;
      alpha	 = ALPHA;
      ret = clSetKernelArg (kernel, 0, sizeof (int), &vectorSize);
      CheckRetValueOfOpenCLFunction (ret);
      ret = clSetKernelArg (kernel, 1, sizeof (float), &alpha);
      CheckRetValueOfOpenCLFunction (ret);
      ret = clSetKernelArg (kernel, 2, sizeof (cl_mem), (void *) &x_dev);
      CheckRetValueOfOpenCLFunction (ret);
      ret = clSetKernelArg (kernel, 3, sizeof (cl_mem), (void *) &y_dev);
      CheckRetValueOfOpenCLFunction (ret);

      /**************************************************************
       *
       * Step 10: Create command queue for device device_ids[device]
       *
       **************************************************************
       */

      command_queue = clCreateCommandQueue (context, device_ids[device],
					    queueProps, &errcode_ret);
      CheckRetValueOfOpenCLFunction (errcode_ret);

      /**************************************************************
       *
       * Step 11: Enqueue a command to execute a kernel
       *
       **************************************************************
       */
      
      localWorkSize  = WORK_ITEMS_PER_WORK_GROUP;
      globalWorkSize = localWorkSize * WORK_GROUPS_PER_NDRANGE;
      start_wall = time (NULL);
      cpu_time   = clock ();
      ret = clEnqueueNDRangeKernel (command_queue, kernel, 1, NULL,
				    &globalWorkSize, &localWorkSize,
				    0, NULL, &profileEvent);
      CheckRetValueOfOpenCLFunction (ret);

      /* finalize							*/
      ret = clFlush (command_queue);
      CheckRetValueOfOpenCLFunction (ret);
      ret = clFinish (command_queue);
      CheckRetValueOfOpenCLFunction (ret);

      /* get times							*/
      cpu_time = clock () - cpu_time;
      end_wall = time (NULL);
      ret = clWaitForEvents (1, &profileEvent);
      CheckRetValueOfOpenCLFunction (ret);
      ret = clGetEventProfilingInfo (profileEvent,
				     CL_PROFILING_COMMAND_START,
				     sizeof (cl_ulong),
				     &clStartTime, NULL);
      CheckRetValueOfOpenCLFunction (ret);
      ret = clGetEventProfilingInfo (profileEvent,
				     CL_PROFILING_COMMAND_END,
				     sizeof (cl_ulong),
				     &clEndTime, NULL);
      CheckRetValueOfOpenCLFunction (ret);

      /**************************************************************
       *
       * Step 12: Read memory objects (result)
       *
       **************************************************************
       */

      ret = clEnqueueReadBuffer (command_queue, y_dev, CL_TRUE, 0,
				 VECTOR_SIZE * sizeof (float), y,
				 0, NULL, NULL);
      CheckRetValueOfOpenCLFunction (ret);


      /* Check result. All elements should have the same value.		*/
      tmp_y0   = y[0];
      tmp_diff = 0;
      for (int i = 0; i < VECTOR_SIZE; ++i)
      {
	if (fabsf (tmp_y0 - y[i]) > EPS)
	{
	  tmp_diff++;
	}
      }
      if (tmp_diff == 0)
      {
	printf ("  Computation was successful. y[0] = %6.2f\n",
		(double) y[0]);
      }
      else
      {
	printf ("  Computation was not successful. %d values differ.\n",
		tmp_diff);
      }

      /* show times							*/
      printf ("  elapsed time      cpu time      GPU elapsed time\n"
	      "      %6.2f s      %6.2f s      %6.2f s\n",
	      difftime (end_wall, start_wall),
	      (double) cpu_time / CLOCKS_PER_SEC,
	      (double) (clEndTime - clStartTime) * 1.0e-9);
      fflush (stdout);


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
      ret = clReleaseMemObject (x_dev);
      CheckRetValueOfOpenCLFunction (ret);
      ret = clReleaseMemObject (y_dev);
      CheckRetValueOfOpenCLFunction (ret);
      ret = clReleaseCommandQueue (command_queue);
      CheckRetValueOfOpenCLFunction (ret);
      ret = clReleaseContext (context);
      CheckRetValueOfOpenCLFunction (ret);
    }
    free (deviceName);
  }
  free (platform_ids);

  return EXIT_SUCCESS;
}
