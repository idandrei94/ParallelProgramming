/* Each C program can be compiled with the NVIDIA C compiler.
 *
 *
 * Compiling:
 *   gcc -o hello_gcc hello.c
 *   nvcc -o hello_nvcc hello.c
 *
 * Running:
 *   ./hello_gcc
 *   ./hello_nvcc
 *
 *
 * File: hello.c			Author: S. Gross
 * Date: 12.09.2016
 *
 */

#include <stdio.h>
#include <stdlib.h>

void hello (void);

int main (void)
{
  printf ("Call my \"hello\" function.\n");
  hello ();
  printf ("Have called my \"hello\" function and terminate now.\n");
  return EXIT_SUCCESS;
}


void hello (void)
{
  printf ("Hello.\n");
}
