/* A small OpenACC program printing some information.
 *
 *
 * Compiling:
 *
 * gcc -fopenacc [-foffload=nvptx-none] -o acc_info_gcc acc_info.c
 * pgcc -acc -ta=nvidia -Minfo=all -o acc_info_pgcc acc_info.c
 *
 *
 * Running:
 *   ./acc_info
 *
 *
 * File: acc_info.c			Author: S. Gross
 * Date: 06.03.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENACC
  #include <openacc.h>
#endif

int main (void)
{
  #ifdef _OPENACC
    #ifdef __GNUC__
      printf ("Supported standard:                _OPENACC = %d\n"
	      "Number of host devices:            %d\n"
	      "Number of none host devices:       %d\n"
	      "Number of attached NVIDIA devices: %d\n",
	      _OPENACC,
	      acc_get_num_devices(acc_device_host),
	      acc_get_num_devices(acc_device_not_host),
	      acc_get_num_devices(acc_device_nvidia));
    #else
      printf ("Supported standard:                       _OPENACC = %d\n"
	      "Number of host devices:                   %d\n"
	      "Number of none host devices:              %d\n"
	      "Number of attached NVIDIA devices:        %d\n"
	      "Number of attached AMD devices:           %d\n"
	      "Number of attached Xeon Phi coprocessors: %d\n"
	      "Number of attached PGI OpenCL devices:    %d\n"
	      "Number of attached NVIDIA OpenCL devices: %d\n"
	      "Number of attached OpenCL devices:        %d\n",
	      _OPENACC,
	      acc_get_num_devices(acc_device_host),
	      acc_get_num_devices(acc_device_not_host),
	      acc_get_num_devices(acc_device_nvidia),
	      acc_get_num_devices(acc_device_radeon),
	      acc_get_num_devices(acc_device_xeonphi),
	      acc_get_num_devices(acc_device_pgi_opencl),
	      acc_get_num_devices(acc_device_nvidia_opencl),
	      acc_get_num_devices(acc_device_opencl));
    #endif
  #else
    printf ("Not compiled for OpenACC.\n");
  #endif
  return EXIT_SUCCESS;
}
