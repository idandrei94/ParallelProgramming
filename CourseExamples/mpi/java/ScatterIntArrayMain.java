/* Small program that distributes an array of integer values with a
 * scatter operation. It needs one process for each element of the
 * matrix.
 *
 * Java uses call-by-value and doesn't support call-by-reference
 * for method parameters with the only exception of object references.
 * Therefore you must use an array with just one element, if you
 * want to send/receive/broadcast/... primitive datatypes.
 *
 * "mpijavac" and Java-bindings are available in "Open MPI
 * version 1.7.4" or newer.
 *
 *
 * Class file generation:
 *   mpijavac ScatterIntArrayMain.java
 *
 * Usage:
 *   mpiexec [parameters] java [parameters] ScatterIntArrayMain
 *
 * Examples:
 *   mpiexec -np 4 java ScatterIntArrayMain
 *   mpiexec -np 4 java -cp $HOME/mpi_classfiles ScatterIntArrayMain
 *
 *
 * File: ScatterIntArrayMain.java	Author: S. Gross
 * Date: 09.09.2013
 *
 */

import mpi.*;

public class ScatterIntArrayMain
{
  static final int SLEEP_FACTOR = 200;	/* 200 ms to get ordered output	*/
  final static int P = 4;		/* # of array elem to scatter	*/

  public static void main (String args[]) throws MPIException,
						 InterruptedException
  {
    int	   mytid,			/* my task id			*/
	   ntasks;			/* number of tasks		*/
    int    intValues[],			/* scatter array of intValues	*/
	   myIntValue[];		/* receive one intValue		*/
    String processorName;		/* name of local machine	*/

    MPI.Init (args);
    mytid  = MPI.COMM_WORLD.getRank ();
    ntasks = MPI.COMM_WORLD.getSize ();
    if (ntasks != P)
    {
      /* wrong number of processes					*/
      if (mytid == 0)
      {
	String className =
	  new ScatterIntArrayMain().getClass().getName();
	System.err.printf ("\nI need %d processes.\n" +
			   "Usage:\n" +
			   "  mpiexec -np %d java %s\n\n",
			   P, P, className);
      }
      MPI.Finalize ();
      System.exit (0);
    }
    processorName = MPI.getProcessorName ();
    intValues	  = new int[P];
    myIntValue    = new int [1];
    if (mytid == 0)
    {
      /* initialize data items						*/
      for (int i = 0; i < P; ++i)
      {
	intValues[i] = i * i;
      }
    }
    /* scatter values to all processes					*/
    MPI.COMM_WORLD.scatter (intValues, 1, MPI.INT,
			    myIntValue, 1, MPI.INT, 0);
    /* Each process prints its received data item. The outputs
     * can intermingle on the screen so that you must use
     * "-output-filename" in Open MPI.
     */
    Thread.sleep (SLEEP_FACTOR * mytid); /* sleep to get ordered output	*/	
    System.out.printf ("\nProcess %d running on %s.\n" +
		       "  myIntValue: %d\n",
		       mytid, processorName, myIntValue[0]);
    MPI.Finalize ();
  }
}
