/* Small program that distributes an array of integer values with a
 * broadcast operation.
 *
 * "mpijavac" and Java-bindings are available in "Open MPI
 * version 1.7.4" or newer.
 *
 *
 * Class file generation:
 *   mpijavac BcastIntArrayMain.java
 *
 * Usage:
 *   mpiexec [parameters] java [parameters] BcastIntArrayMain
 *
 * Examples:
 *   mpiexec -np 2 java BcastIntArrayMain
 *   mpiexec -np 2 java -cp $HOME/mpi_classfiles BcastIntArrayMain
 *
 *
 * File: BcastIntArrayMain.java		Author: S. Gross
 * Date: 12.09.2013
 *
 */

import mpi.*;

public class BcastIntArrayMain
{
  static final int SLEEP_FACTOR = 200;	/* 200 ms to get ordered output	*/
  static final int ELEM_PER_LINE = 4;	/* # of array elements per line	*/
  static final int ARRAY_SIZE = 4;

  public static void main (String args[]) throws MPIException,
						 InterruptedException
  {
    int	   mytid;			/* my task id			*/
    int    intValues[];			/* broadcast array of intValues	*/
    String processorName;		/* name of local machine	*/

    MPI.Init (args);
    processorName = MPI.getProcessorName ();
    mytid	   = MPI.COMM_WORLD.getRank ();
    intValues	   = new int[ARRAY_SIZE];
    if (mytid == 0)
    {
      /* initialize data items						*/
      for (int i = 0; i < ARRAY_SIZE; ++i)
      {
       intValues[i] = i * 11;
      }
    }
    /* broadcast value to all processes					*/
    MPI.COMM_WORLD.bcast (intValues, ARRAY_SIZE, MPI.INT, 0);
    /* Each process prints its received data item. The outputs
     * can intermingle on the screen so that you must use
     * "-output-filename" in Open MPI.
     */
    Thread.sleep (SLEEP_FACTOR * mytid); /* sleep to get ordered output	*/	
    System.out.printf ("\nProcess %d running on %s.\n",
		       mytid, processorName);
    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
      System.out.printf ("  intValues[%d]: %d", i, intValues[i]);
      if ((i + 1) % ELEM_PER_LINE == 0)
      {
	System.out.printf ("\n");
      }
    }
    MPI.Finalize ();
  }
}
