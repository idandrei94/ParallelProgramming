/* The program demonstrates how to handle exceptions. This version
 * triggers a Java exception in an MPI function.
 *
 * "mpijavac" and Java-bindings are available in "Open MPI
 * version 1.7.4" or newer.
 *
 *
 * Class file generation:
 *   mpijavac ExceptionJavaMain.java
 *
 * Usage:
 *   mpiexec [parameters] java [parameters] ExceptionJavaMain
 *
 * Examples:
 *   mpiexec java ExceptionJavaMain
 *   mpiexec java -cp $HOME/mpi_classfiles ExceptionJavaMain
 *
 *
 * File: ExceptionJavaMain.java		Author: S. Gross
 * Date: 09.09.2016
 *
 */

import mpi.*;

public class ExceptionJavaMain
{
  static final int SLEEP_FACTOR = 200;	/* 200 ms to get ordered output	*/

  public static void main (String args[]) throws MPIException,
						 InterruptedException
  {
    int mytid,				/* my task id			*/
	intValue[] = new int[1];	/* broadcast one intValue	*/

    MPI.Init(args);
    mytid = MPI.COMM_WORLD.getRank ();
    if (mytid == 0)
    {
      intValue[0] = 10;			/* arbitrary value		*/
    }

    /* Each process prints a short message. The outputs can intermingle
     * on the screen so that you must use "-output-filename" with Open
     * MPI or let each process sleep a different short time.
     */
    Thread.sleep (SLEEP_FACTOR * mytid); /* sleep to get ordered output	*/	
    /* handle errors myself						*/
    System.out.printf ("Process %d: Set error handler for " +
		       "MPI.COMM_WORLD to MPI.ERRORS_RETURN.\n", mytid);
    MPI.COMM_WORLD.setErrhandler (MPI.ERRORS_RETURN);
    try
    {
      /* use index out-of bounds to produce an error			*/
      System.out.printf ("Call \"bcast\" with index out-of bounds.\n");
      MPI.COMM_WORLD.bcast (intValue, 2, MPI.INT, 0);
    }
    catch (MPIException e)
    {
      Thread.sleep (SLEEP_FACTOR * mytid); /* sleep for ordered output	*/	
      System.err.printf ("  Process %2d: Error class: %d\n" +
			 "              Error code:  %d\n" +
			 "              Error message: %s\n",
			 mytid, e.getErrorClass(), e.getErrorCode(),
			 e.getMessage ());
      MPI.Finalize ();
      System.exit (0);
    }
    catch (IndexOutOfBoundsException e)
    {
      /* Try to find the root cause for the exception			*/
      Throwable cause = null; 
      Throwable result = e;

      while (null != (cause = result.getCause()) && (result != cause))
      {
        result = cause;
      }

      Thread.sleep (SLEEP_FACTOR * mytid); /* sleep for ordered output	*/	
      System.err.printf ("  Process %2d: Error message: %s\n",
			 mytid, result);
    }
    /* reinstall default error handler				*/
    MPI.COMM_WORLD.setErrhandler (MPI.ERRORS_ARE_FATAL);
    MPI.Finalize ();
  }
}
