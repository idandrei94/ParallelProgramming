/* Small program that distributes a double value with a
 * broadcast operation.
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
 *   mpijavac BcastDoubleMain.java
 *
 * Usage:
 *   mpiexec [parameters] java [parameters] BcastDoubleMain
 *
 * Examples:
 *   mpiexec -np 2 java BcastDoubleMain
 *   mpiexec -np 2 java -cp $HOME/mpi_classfiles BcastDoubleMain
 *
 *
 * File: BcastDoubleMain.java		Author: S. Gross
 * Date: 09.09.2013
 *
 */

import mpi.*;

public class BcastDoubleMain
{
  static final int SLEEP_FACTOR = 200;	/* 200 ms to get ordered output	*/

  public static void main (String args[]) throws MPIException,
						 InterruptedException
  {
    int	   mytid;			/* my task id			*/
    double doubleValue[];		/* broadcast one doubleValue	*/
    String processorName;		/* name of local machine	*/

    MPI.Init (args);
    processorName  = MPI.getProcessorName ();
    mytid	   = MPI.COMM_WORLD.getRank ();
    doubleValue    = new double[1];
    doubleValue[0] = -1.0;
    if (mytid == 0)
    {
      /* initialize data item						*/
      doubleValue[0] = 1234567.0;
    }
    /* broadcast value to all processes					*/
    MPI.COMM_WORLD.bcast (doubleValue, 1, MPI.DOUBLE, 0);
    /* Each process prints its received data item. The outputs
     * can intermingle on the screen so that you must use
     * "-output-filename" in Open MPI.
     */
    Thread.sleep (SLEEP_FACTOR * mytid); /* sleep to get ordered output	*/	
    System.out.printf ("\nProcess %d running on %s.\n" +
		       "  doubleValue: %f\n",
		       mytid, processorName, doubleValue[0]);
    MPI.Finalize ();
  }
}
