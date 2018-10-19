/* Print some properties for all available OpenCL devices.
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
 *
 *
 * Compiling:
 *   Linux:
 *     gcc -o platformInfo platformInfo.c errorCodes.c -lOpenCL
 *   Mac OS X:
 *	 Uses frameworks:
 *	   -framework OpenCL
 *	   -framework OpenGL -framework GLUT
 *     gcc -o platformInfo platformInfo.c errorCodes.c \
 *	   -framework OpenCL
 *   Windows:
 *     cl platformInfo.c errorCodes.c OpenCL.lib
 *
 * Running:
 *   ./platformInfo
 *
 *
 * File: platformInfo.c			Author: S. Gross
 * Date: 07.06.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef __APPLE__
  #include <CL/cl.h>
#else
  #include <OpenCL/opencl.h>
#endif


#define MAX_STRING_LENGTH 256


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


/* Platform information table
 *
 * "padding" is necessary to avoid the warning "padding size of
 * 'struct ...' with 4 bytes to alignment boundary" from "clang".
 */
static const struct { const char * const paramText;
		      const cl_platform_info param_name;
		      const char padding[4];
} platformInfoTable [] =
  { {"  Platform name:    ", CL_PLATFORM_NAME, ""},
    {"  Platform version: ", CL_PLATFORM_VERSION, ""},
    {"  Platform profile: ", CL_PLATFORM_PROFILE, ""}
  };


/* Device information table for parameters with a "char *" return value
 *
 * "padding" is necessary to avoid the warning "padding size of
 * 'struct ...' with 4 bytes to alignment boundary" from "clang".
 */
static const struct { const char * const paramText;
		      const cl_device_info param_name;
		      const char padding[4];
} deviceInfoCharTable [] =
  { {"    Device name:                  ", CL_DEVICE_NAME, ""},
    {"    OpenCL device version:        ", CL_DEVICE_VERSION, ""}
  };


/* Device information table for parameters with a "cl_device_type"
 * return value
 *
 * "padding" is necessary to avoid the warning "padding size of
 * 'struct ...' with 4 bytes to alignment boundary" from "clang".
 */
static const struct { const char * const paramText;
		      const cl_device_info param_name;
		      const char padding[4];
} deviceInfoDevTypeTable [] =
  { {"    Device type:                  ", CL_DEVICE_TYPE, ""}
  };


/* Table for device type bitfields					*/
static const struct { const char * const bitfieldText;
		      const cl_device_type bitfieldName;
} deviceTypeBitfieldTable [] =
  { {"CL_DEVICE_TYPE_CPU",	   CL_DEVICE_TYPE_CPU},
    {"CL_DEVICE_TYPE_GPU",	   CL_DEVICE_TYPE_GPU},
    {"CL_DEVICE_TYPE_ACCELERATOR", CL_DEVICE_TYPE_ACCELERATOR},
    {"CL_DEVICE_TYPE_DEFAULT",	   CL_DEVICE_TYPE_DEFAULT}
  };


/* Device information table for parameters with an "uint" return value
 *
 * "padding" is necessary to avoid the warning "padding size of
 * 'struct ...' with 4 bytes to alignment boundary" from "clang".
 */
static const struct { const char * const paramText;
		      const cl_device_info param_name;
		      const char padding[4];
} deviceInfoUintTable [] =
  { {"    Max. compute units:           ",
       CL_DEVICE_MAX_COMPUTE_UNITS, ""},
    {"    Max. work-item dimensions:    ",
       CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, ""},
    {"    Global memory cacheline size: ",
       CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, ""}
  };


/* Device information table for parameters with an "ulong" return value
 *
 * "padding" is necessary to avoid the warning "padding size of
 * 'struct ...' with 4 bytes to alignment boundary" from "clang".
 */
static const struct { const char * const paramText;
		      const cl_device_info param_name;
		      const char padding[4];
} deviceInfoUlongTable [] =
  { {"    Global memory cache size:     ",
       CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, ""},
    {"    Global memory size:           ",
       CL_DEVICE_GLOBAL_MEM_SIZE, ""},
    {"    Max. constant buffer size:    ",
       CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, ""},
    {"    Local memory size:            ",
       CL_DEVICE_LOCAL_MEM_SIZE, ""}
  };


/* Device information table for parameters with a "size_t" return value
 *
 * "padding" is necessary to avoid the warning "padding size of
 * 'struct ...' with 4 bytes to alignment boundary" from "clang".
 */
static const struct { const char * const paramText;
		      const cl_device_info param_name;
		      const char padding[4];
} deviceInfoSizetTable [] =
  { {"    Max. work-group size:         ",
       CL_DEVICE_MAX_WORK_GROUP_SIZE, ""}
  };


/* Device information table for parameters with a "bool" return value
 *
 * "padding" is necessary to avoid the warning "padding size of
 * 'struct ...' with 4 bytes to alignment boundary" from "clang".
 */
static const struct { const char * const paramText;
		      const cl_device_info param_name;
		      const char padding[4];
} deviceInfoBoolTable [] =
  { {"    Image support:                ",
       CL_DEVICE_IMAGE_SUPPORT, ""},
    {"    Device compiler available:    ",
       CL_DEVICE_COMPILER_AVAILABLE, ""}
  };



int main (void)
{
  int	  platformInfoTableSize,
	  deviceInfoTableSize;
  cl_uint numPlatforms;			/* # of available platforms	*/
  cl_int  ret;				/* return value of a function	*/
  cl_platform_id *platform_ids;

  platformInfoTableSize =
    sizeof (platformInfoTable) / sizeof (platformInfoTable[0]);

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

  /* get and print some information for all platforms			*/
  for (unsigned int platform = 0; platform < numPlatforms; ++platform)
  {
    char	 *paramValue;
    size_t	 paramValueSize;
    cl_uint	 numDevices;
    cl_device_id *device_ids;

    /************************************************************
     *
     *                        Platform Info
     *
     ************************************************************
     */
    printf ("\nSome data about platform %d\n", platform);
    for (int i = 0; i < platformInfoTableSize; ++i)
    {
      ret = clGetPlatformInfo (platform_ids[platform],
			       platformInfoTable[i].param_name,
			       0, NULL, &paramValueSize);
      CheckRetValueOfOpenCLFunction (ret);
      paramValue = (char *) malloc (paramValueSize * sizeof (char));
      TestEqualsNULL (paramValue);
      ret = clGetPlatformInfo (platform_ids[platform],
			       platformInfoTable[i].param_name,
			       paramValueSize, paramValue,
			       NULL);
      CheckRetValueOfOpenCLFunction (ret);
      printf ("%s%s\n", platformInfoTable[i].paramText,
	      paramValue);
      free (paramValue);
    }
 
    /************************************************************
     *
     *                        Device Info
     *
     ************************************************************
     */
    
    /* get the number of available devices				*/
    ret = clGetDeviceIDs (platform_ids[platform], CL_DEVICE_TYPE_ALL,
			  0, NULL, &numDevices);
    CheckRetValueOfOpenCLFunction (ret);
    if (numDevices > 0)
    {
      printf ("\n  Found %d device(s).\n", numDevices);
    }
    else
    {
      printf ("\n  Couldn't find any OpenCL capable devices.\n\n");
      break;				/* try next platform		*/
    }

    /* get device IDs							*/
    device_ids = (cl_device_id *) malloc (numDevices *
					  sizeof (cl_device_id));
    TestEqualsNULL (device_ids);
    ret = clGetDeviceIDs (platform_ids[platform], CL_DEVICE_TYPE_ALL,
			  numDevices, device_ids, NULL);
    CheckRetValueOfOpenCLFunction (ret);
   
    /* get and print some information for all devices			*/
    for (unsigned int device = 0; device < numDevices; ++device)
    {
      char	*charParamValue;
      cl_uint	uintParamValue;
      cl_ulong	ulongParamValue;
      cl_bool	boolParamValue;
      size_t	sizetParamValue;
      cl_device_type devTypeParamValue;

      /* Device parameters with a "char *" return value			*/
      deviceInfoTableSize =
	sizeof (deviceInfoCharTable) / sizeof (deviceInfoCharTable[0]);
      for (int i = 0; i < deviceInfoTableSize; ++i)
      {
	printf ("\n");
	/* get string length						*/
	ret = clGetDeviceInfo (device_ids[device],
			       deviceInfoCharTable[i].param_name,
			       0, NULL, &paramValueSize);
	CheckRetValueOfOpenCLFunction (ret);
	charParamValue =
	  (char *) malloc (paramValueSize * sizeof (char));
	TestEqualsNULL (charParamValue);
	ret = clGetDeviceInfo (device_ids[device],
			       deviceInfoCharTable[i].param_name,
			       paramValueSize, charParamValue,
			       NULL);
	CheckRetValueOfOpenCLFunction (ret);
	printf ("%s%s\n", deviceInfoCharTable[i].paramText,
		charParamValue);
	free (charParamValue);
      }

      /* Device parameters with a "cl_device_type" return value	*/
      deviceInfoTableSize = sizeof (deviceInfoDevTypeTable) /
	sizeof (deviceInfoDevTypeTable[0]);
      for (int i = 0; i < deviceInfoTableSize; ++i)
      {
	char devType[MAX_STRING_LENGTH];
	int  deviceTypeBitfieldTableSize;

	ret = clGetDeviceInfo (device_ids[device],
			       deviceInfoDevTypeTable[i].param_name,
			       sizeof (cl_device_type),
			       &devTypeParamValue, NULL);
	CheckRetValueOfOpenCLFunction (ret);

	deviceTypeBitfieldTableSize =
	  sizeof (deviceTypeBitfieldTable) /
	  sizeof (deviceTypeBitfieldTable[0]);
	/* empty string buffer						*/
	memset (devType, '\0', MAX_STRING_LENGTH);
	for (int j = 0; j < deviceTypeBitfieldTableSize; ++j)
	{
	  if ((devTypeParamValue &
	       deviceTypeBitfieldTable[j].bitfieldName) != 0)
	  {
	    if (strlen (devType) > 0)
	    {
	      strncat (devType, " | ",
		       MAX_STRING_LENGTH - strlen (devType));
	    }
	    strncat (devType, deviceTypeBitfieldTable[j].bitfieldText,
		     MAX_STRING_LENGTH - strlen (devType));
	  }
	}
	printf ("%s%s\n", deviceInfoDevTypeTable[i].paramText,
		devType);
      }

      /* Device parameters with an "uint" return value			*/
      deviceInfoTableSize =
	sizeof (deviceInfoUintTable) / sizeof (deviceInfoUintTable[0]);
      for (int i = 0; i < deviceInfoTableSize; ++i)
      {
	ret = clGetDeviceInfo (device_ids[device],
			       deviceInfoUintTable[i].param_name,
			       sizeof (cl_uint), &uintParamValue,
			       NULL);
	CheckRetValueOfOpenCLFunction (ret);
	printf ("%s%u\n", deviceInfoUintTable[i].paramText,
		uintParamValue);
      }

      /* Device parameters with an "ulong" return value			*/
      deviceInfoTableSize =
	sizeof (deviceInfoUlongTable) / sizeof (deviceInfoUlongTable[0]);
      for (int i = 0; i < deviceInfoTableSize; ++i)
      {
	ret = clGetDeviceInfo (device_ids[device],
			       deviceInfoUlongTable[i].param_name,
			       sizeof (cl_ulong), &ulongParamValue,
			       NULL);
	CheckRetValueOfOpenCLFunction (ret);
	printf ("%s%llu\n", deviceInfoUlongTable[i].paramText,
		(long long unsigned int) ulongParamValue);
      }

      /* Device parameters with a "size_t" return value			*/
      deviceInfoTableSize =
	sizeof (deviceInfoSizetTable) / sizeof (deviceInfoSizetTable[0]);
      for (int i = 0; i < deviceInfoTableSize; ++i)
      {
	ret = clGetDeviceInfo (device_ids[device],
			       deviceInfoSizetTable[i].param_name,
			       sizeof (size_t), &sizetParamValue,
			       NULL);
	CheckRetValueOfOpenCLFunction (ret);

	/* "sizetParamValue" has type "size_t" so that "%zu"
	 * would be the favoured format specifier in a
	 * printf-statement. Unfortunately, Microsoft Visual
	 * Studio doesn't support "%zu" with older versions,
	 * so that "%llu" is a better and portable choice.
	 */
	printf ("%s%llu\n", deviceInfoSizetTable[i].paramText,
		(long long unsigned int) sizetParamValue);
      }

      /* Device parameters for CL_DEVICE_MAX_WORK_GROUP_SIZE		*/
      {
	cl_uint workItemDim;
	size_t *workItemSizes;
	char *paramName = "    Max. work-item sizes:         ";
	
	ret = clGetDeviceInfo (device_ids[device],
			       CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
			       sizeof (cl_uint), &workItemDim, NULL);
	CheckRetValueOfOpenCLFunction (ret);
	workItemSizes = (size_t *) malloc (workItemDim * sizeof (size_t));
	TestEqualsNULL (workItemSizes);
	ret = clGetDeviceInfo (device_ids[device],
			       CL_DEVICE_MAX_WORK_ITEM_SIZES,
			       workItemDim * sizeof (size_t),
			       workItemSizes, NULL);
	CheckRetValueOfOpenCLFunction (ret);
	printf ("%s(", paramName);
	for (unsigned int i = 0; i < workItemDim - 1; ++i)
	{
	  /* "workItemSizes[]" have type "size_t" so that "%zu"
	   * would be the favoured format specifier in a
	   * printf-statement. Unfortunately, Microsoft Visual
	   * Studio doesn't support "%zu" with older versions,
	   * so that "%llu" is a better and portable choice.
	   */
	  printf ("%llu, ", (long long unsigned int) workItemSizes[i]);
	}
	printf ("%llu)\n",
		(long long unsigned int) workItemSizes[workItemDim - 1]);
	free (workItemSizes);
      }

      /* Device parameters with a "cl_bool" return value		*/
      deviceInfoTableSize =
	sizeof (deviceInfoBoolTable) / sizeof (deviceInfoBoolTable[0]);
      for (int i = 0; i < deviceInfoTableSize; ++i)
      {
	ret = clGetDeviceInfo (device_ids[device],
			       deviceInfoBoolTable[i].param_name,
			       sizeof (cl_bool), &boolParamValue,
			       NULL);
	CheckRetValueOfOpenCLFunction (ret);
	printf ("%s%s\n", deviceInfoBoolTable[i].paramText,
		(boolParamValue == CL_TRUE) ? "yes" : "no");
      }
    }
    free (device_ids);
  }
  free (platform_ids);
  return EXIT_SUCCESS;
}
