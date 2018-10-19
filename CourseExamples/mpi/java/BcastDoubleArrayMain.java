/* Small program that distributes an array of double values with a
 * broadcast operation.
 *
 * "mpijavac" and Java-bindings are available in "Open MPI
 * version 1.7.4" or newer.
 *
 *
 * Class file generation:
 *   mpijavac BcastDoubleArrayMain.java
 *
 * Usage:
 *   mpiexec [parameters] java [parameters] BcastDoubleArrayMain
 *
 * Examples:
 *   mpiexec -np 2 java BcastDoubleArrayMain
 *   mpiexec -np 2 java -cp $HOME/mpi_classfiles BcastDoubleArrayMain
 *
 *
 * File: BcastDoubleArrayMain.java	Author: S. Gross
 * Date: 12.09.2013
 *
 */

import mpi.*;

public class BcastDoubleArrayMain
{
  static final int SLEEP_FACTOR = 200;	/* 200 ms to get ordered output	*/
  static final int ELEM_PER_LINE = 2;	/* # of array elements per line	*/
  static final int ARRAY_SIZE = 4;

  public static void main (String args[]) throws MPIException,
						 InterruptedException
  {
    int	   mytid;			/* my task id			*/
    double doubleValues[];		/* bcast array of doubleValues	*/
    String processorName;		/* name of local machine	*/

    MPI.Init (args);
    processorName = MPI.getProcessorName ();
    mytid	   = MPI.COMM_WORLD.getRank ();
    doubleValues   = new double[ARRAY_SIZE];
    if (mytid == 0)
    {
      /* initialize data items						*/
      for (int i = 0; i < ARRAY_SIZE; ++i)
      {
       doubleValues[i] = i * 1.1;
      }
    }
    /* broadcast value to all processes					*/
    MPI.COMM_WORLD.bcast (doubleValues, ARRAY_SIZE, MPI.DOUBLE, 0);
    /* Each process prints its received data item. The outputs
     * can intermingle on the screen so that you must use
     * "-output-filename" in Open MPI.
     */
    Thread.sleep (SLEEP_FACTOR * mytid); /* sleep to get ordered output	*/	
    System.out.printf ("\nProcess %d running on %s.\n",
		       mytid, processorName);
    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
      System.out.printf ("  doubleValues[%d]: %f", i, doubleValues[i]);
      if ((i + 1) % ELEM_PER_LINE == 0)
      {
	System.out.printf ("\n");
      }
    }
    MPI.Finalize ();
  }
}
