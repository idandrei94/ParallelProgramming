/* This program prints the size, minimum and maximum value, and the
 * number of digits of the mantissa for floating-point types.
 *
 * Compiling:
 *   cc  -o Floatingpoint Floatingpoint.c
 *   gcc -o Floatingpoint Floatingpoint.c
 *   icc -o Floatingpoint Floatingpoint.c
 *   cl  /Ox /GL Floatingpoint.c
 *
 * Running:
 *   ./Floatingpoint
 *
 *
 * File: Floatingpoint.c		Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

int main (void)
{
  float       f_min = FLT_MIN, f_max = FLT_MAX;
  double      d_min = DBL_MIN, d_max = DBL_MAX;
  long double ld_min = LDBL_MIN, ld_max = LDBL_MAX;
  int	      f_mant_dig = FLT_MANT_DIG, d_mant_dig = DBL_MANT_DIG,
	      ld_mant_dig = LDBL_MANT_DIG;

  printf ("float:\n"
	  "  size (bytes)        %d\n"
	  "  min value:          %e\n"
	  "  max value:          %e\n"
	  "  mantissa (digits):  %d\n"
	  "double:\n"
	  "  size (bytes)        %d\n"
	  "  min value:          %e\n"
	  "  max value:          %e\n"
	  "  mantissa (digits):  %d\n"
	  "long double:\n"
	  "  size (bytes)        %d\n"
	  "  min value:          %Le\n"
	  "  max value:          %Le\n"
	  "  mantissa (digits):  %d\n",
	  (int) sizeof (float), (double) f_min, (double) f_max,
	  f_mant_dig,
	  (int) sizeof (double), d_min, d_max, d_mant_dig,
	  (int) sizeof (long double), ld_min, ld_max, ld_mant_dig);
  return EXIT_SUCCESS;
}
