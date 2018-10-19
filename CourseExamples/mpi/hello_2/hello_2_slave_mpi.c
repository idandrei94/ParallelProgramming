/* Another MPI-version of the "hello world" program, which delivers
 * some information about its machine and operating system. In this
 * version the functions "master" and "slave" from "hello_1_mpi.c"
 * are implemented as independant processes. This is the file for the
 * "slave".
 *
 * To simulate "real work" the "slave"-process waits up to MAX_WTIME
 * seconds before replying to a message request.
 *
 *
 * Compiling:
 *   mpicc -o hello_2_mpi hello_2_mpi.c -lm
 *   mpicc -o hello_2_slave_mpi hello_2_slave_mpi.c -lm
 *
 * Running:
 *   mpiexec -np 1 hello_2_mpi : -np <number of processes> \
 *	     --host <hostname(s)> hello_2_slave_mpi
 *
 *
 * File: hello_2_slave.c	       	Author: S. Gross
 * Date: 16.07.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/utsname.h>
#include "mpi.h"

#define	BUF_SIZE	255		/* message buffer size		*/
#define	MAX_TASKS	12		/* max. number of tasks		*/
#define	SENDTAG		1		/* send message command		*/
#define	EXITTAG		2		/* termination command		*/
#define	MSGTAG		3		/* normal message token		*/

#define ENTASKS		-1		/* error: too many tasks	*/

int main (int argc, char *argv[])
{
  struct utsname sys_info;		/* system information		*/
  int  mytid,				/* my task id			*/
       ntasks,				/* number of parallel tasks	*/
       namelen,				/* length of processor name	*/
       more_to_do;
  char processor_name[MPI_MAX_PROCESSOR_NAME],
       buf[BUF_SIZE];			/* message buffer      		*/
  MPI_Status stat;			/* message details		*/

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &mytid);
  MPI_Comm_size (MPI_COMM_WORLD, &ntasks);
  MPI_Get_processor_name (processor_name, &namelen);
  /* With the next statement every process executing this code will
   * print one line on the display. It may happen that the lines will
   * get mixed up because the display is a critical section. In general
   * only one process (mostly the process with rank 0) will print on
   * the display and all other processes will send their messages to
   * this process. Nevertheless for debugging purposes (or to
   * demonstrate that it is possible) it may be useful if every
   * process prints itself.
   */
  fprintf (stdout, "Process %d of %d running on %s\n",
	   mytid, ntasks, processor_name);
  fflush (stdout);
  MPI_Barrier (MPI_COMM_WORLD);		/* wait for all other processes	*/

  srand ((unsigned int) (time ((time_t *) NULL) * mytid * mytid));
  more_to_do = 1;
  while (more_to_do == 1)
  {
    /* wait for a message from the master task				*/
    MPI_Recv (buf, BUF_SIZE, MPI_CHAR, 0, MPI_ANY_TAG,
	      MPI_COMM_WORLD, &stat);
    if (stat.MPI_TAG != EXITTAG)
    {
      uname (&sys_info);
      strcpy (buf, "\n    hostname:          ");
      strncpy (buf + strlen (buf), sys_info.nodename,
	       BUF_SIZE - strlen (buf));
      strncpy (buf + strlen (buf), "\n    operating system:  ",
	       BUF_SIZE - strlen (buf));
      strncpy (buf + strlen (buf), sys_info.sysname,
	       BUF_SIZE - strlen (buf));
      strncpy (buf + strlen (buf), "\n    release:           ",
	       BUF_SIZE - strlen (buf));
      strncpy (buf + strlen (buf), sys_info.release,
	       BUF_SIZE - strlen (buf));
      strncpy (buf + strlen (buf), "\n    processor:         ",
	       BUF_SIZE - strlen (buf));
      strncpy (buf + strlen (buf), sys_info.machine,
	       BUF_SIZE - strlen (buf));
      MPI_Send (buf, (int) strlen (buf), MPI_CHAR, stat.MPI_SOURCE,
		MSGTAG, MPI_COMM_WORLD);
    }
    else
    {
      more_to_do = 0;			/* terminate			*/
    }
  }
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
