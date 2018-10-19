/* Small program that distributes an array of double values with a
 * scatter operation. It needs one process for each element of the
 * array.
 *
 * At the moment (January 2013) the Java interface of Open MPI
 * for the scatter operation can only handle object arrays so
 * that you must define an array even for a single object.
 *
 * "mpijavac" and Java-bindings are available in "Open MPI
 * version 1.7.4" or newer.
 *
 *
 * Class file generation:
 *   mpijavac ScatterDoubleArrayMain.java
 *
 * Usage:
 *   mpiexec [parameters] java [parameters] ScatterDoubleArrayMain
 *
 * Examples:
 *   mpiexec -np 4 java ScatterDoubleArrayMain
 *   mpiexec -np 4 java -cp $HOME/mpi_classfiles ScatterDoubleArrayMain
 *
 *
 * File: ScatterDoubleArrayMain.java	Author: S. Gross
 * Date: 09.09.2013
 *
 */

import mpi.*;

public class ScatterDoubleArrayMain
{
  static final int SLEEP_FACTOR = 200;	/* 200 ms to get ordered output	*/
  final static int P = 4;		/* # of array elem to scatter	*/

  public static void main (String args[]) throws MPIException,
						 InterruptedException
  {
    int	   mytid,			/* my task id			*/
	   ntasks;			/* number of tasks		*/
    double doubleValues[],		/* scatter array of doubleValues*/
	   myDoubleValue[];		/* array with one element	*/
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
	  new ScatterDoubleArrayMain().getClass().getName();
	System.err.printf ("\nI need %d processes.\n" +
			   "Usage:\n" +
			   "  mpiexec -np %d java %s\n\n",
			   P, P, className);
      }
      MPI.Finalize ();
      System.exit (0);
    }
    processorName = MPI.getProcessorName ();
    doubleValues  = new double[P];
    if (mytid == 0)
    {
      /* initialize data items						*/
      for (int i = 0; i < P; ++i)
      {
	doubleValues[i] = i * i;
      }
    }
    /* At the moment the Java interface of Open MPI can only handle
     * object arrays so that you must define an array even for a
     * single object.
     */
    myDoubleValue = new double [1];
    /* scatter values to all processes					*/
    MPI.COMM_WORLD.scatter (doubleValues, 1, MPI.DOUBLE,
			    myDoubleValue, 1, MPI.DOUBLE, 0);
    /* Each process prints its received data item. The outputs
     * can intermingle on the screen so that you must use
     * "-output-filename" in Open MPI.
     */
    Thread.sleep (SLEEP_FACTOR * mytid); /* sleep to get ordered output	*/	
    System.out.printf ("\nProcess %d running on %s.\n" +
		       "  myDoubleValue: %.2f\n",
		       mytid, processorName, myDoubleValue[0]);
    MPI.Finalize ();
  }
}
