/* A very small MPI Java program which prints a message.
 * "mpijavac" and Java-bindings are available in "Open MPI
 * version 1.7.4" or newer.
 *
 *
 * Class file generation:
 *   mpijavac HelloMainWithBarrier.java
 *
 * Usage:
 *   mpiexec [parameters] java [parameters] HelloMainWithBarrier
 *
 * Examples:
 *   mpiexec java HelloMainWithBarrier
 *   mpiexec java -cp $HOME/mpi_classfiles HelloMainWithBarrier
 *   mpiexec -np 4 java HelloMainWithBarrier
 *   mpiexec -host tyr,linpc1,sunpc1 -np 4 java HelloMainWithBarrier
 *
 *
 * File: HelloMainWithBarrier.java	Author: S. Gross
 * Date: 09.09.2013
 *
 */

import mpi.*;

public class HelloMainWithBarrier
{
  static final int SLEEP_FACTOR = 200;	/* 200 ms to get ordered output	*/

  public static void main (String args[]) throws MPIException,
						 InterruptedException
  {
    int  mytid,				/* my task id			*/
	 ntasks;			/* number of parallel tasks	*/
    String processorName;		/* name of local machine	*/

    MPI.Init (args);
    mytid  = MPI.COMM_WORLD.getRank ();
    ntasks = MPI.COMM_WORLD.getSize ();
    processorName = MPI.getProcessorName ();
    /* With the next statement every process executing this code
     * will print one line on the display. It may happen that the
     * lines will get mixed up because the display is a critical
     * section. In general only one process (mostly the process with
     * rank 0) will print on the display and all other processes
     * will send their messages to this process. Nevertheless for
     * debugging purposes (or to demonstrate that it is possible)
     * it may be useful if every process prints itself.
     */
    Thread.sleep (SLEEP_FACTOR * mytid); /* sleep to get ordered output	*/	
    System.out.println ("Process " + mytid + " of " + ntasks +
			" running on " + processorName);
    MPI.COMM_WORLD.barrier ();		/* wait for all other processes	*/
    System.out.println ("Done! :-)) ");
    MPI.Finalize ();
  }
}
