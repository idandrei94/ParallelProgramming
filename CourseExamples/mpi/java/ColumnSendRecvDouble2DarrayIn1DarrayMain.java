/* Small program that creates and prints column vectors of a matrix.
 * This version uses Send/Recv operations and needs one process
 * for each column of the matrix and an additional coordinator
 * process.
 *
 * "Datatype.Vector" needs a matrix in a contiguous memory area
 * (January 2013). Unfortunately Java stores a 2-dimensional array
 * as array of arrays (one 1-dimensional array for each row and
 * another 1-dimensional array for the object IDs of the rows),
 * i.e., each 1-dimensional array is stored in a contiguous memory
 * area, but the whole matrix isn't stored in a contiguous memory
 * area. Therefore it is necessary to simulates a 2-dimensional
 * array in an 1-dimensional array and to compute all indices
 * manually at the moment.
 *
 * "mpijavac" and Java-bindings are available in "Open MPI
 * version 1.7.4" or newer.
 *
 *
 * Class file generation:
 *   mpijavac ColumnSendRecvDouble2DarrayIn1DarrayMain.java
 *
 * Usage:
 *   mpiexec [parameters] java [parameters] \
 *	ColumnSendRecvDouble2DarrayIn1DarrayMain
 *
 * Examples:
 *   mpiexec -np 7 java ColumnSendRecvDouble2DarrayIn1DarrayMain
 *   mpiexec -np 7 java -cp $HOME/mpi_classfiles \
 *	ColumnSendRecvDouble2DarrayIn1DarrayMain
 *
 *
 * File: ColumnSendRecvDouble2DarrayIn1DarrayMain.java	Author: S. Gross
 * Date: 10.09.2013
 *
 */

import java.nio.*;
import mpi.*;
import static mpi.MPI.slice;

public class ColumnSendRecvDouble2DarrayIn1DarrayMain
{
  static final int SLEEP_FACTOR = 200;	/* 200 ms to get ordered output	*/
  static final int P = 4;			/* # of rows		*/
  static final int Q = 6;			/* # of columns		*/
  static final int NUM_ELEM_PER_LINE = 6;	/* to print a vector	*/
  static final int SIZEOF_DOUBLE = 8;

  public static void main (String args[]) throws MPIException,
						 InterruptedException
  {
    int      ntasks,			/* number of parallel tasks	*/
	     mytid,			/* my task id			*/
	     i, j,			/* loop variables		*/
	     tmp;			/* temporary value		*/
    /* We need "&matrix[0][i]" to send the i-th column of "matrix" to
     * process "i" so that we must use a DoubleBuffer instead of an
     * array for the send-operation.
     */
    DoubleBuffer matrixBuffer;
    //    double matrix[][] = new double[P][Q],
    double   matrix[] = new double[P * Q],
	     column[] = new double[P];
    Datatype tmp_column_t, column_t;	/* strided vector		*/

    MPI.Init (args);
    matrixBuffer = MPI.newDoubleBuffer(P * Q);
    mytid  = MPI.COMM_WORLD.getRank ();
    ntasks = MPI.COMM_WORLD.getSize ();
    if (ntasks != (Q + 1))
    {
      /* wrong number of processes					*/
      if (mytid == 0)
      {
	String className =
	  new ColumnSendRecvDouble2DarrayIn1DarrayMain().getClass().getName();
	System.err.printf ("\nI need %d processes.\n" +
			   "Usage:\n" +
			   "  mpiexec -np %d java %s\n\n",
			   Q + 1, Q +  1, className);
      }
      MPI.Finalize ();
      System.exit (0);
    }
    /* Build the new type for a strided vector.				*/
    tmp_column_t = Datatype.createVector (P, 1, Q, MPI.DOUBLE);
    column_t     = Datatype.createResized (tmp_column_t, 0, SIZEOF_DOUBLE);
    column_t.commit ();
    tmp_column_t.free ();
    if (mytid == 0)
    {
      tmp = 1;
      for (i = 0; i < P; ++i)			/* initialize matrix	*/
      {
	for (j = 0; j < Q; ++j)
        {
	  //	  matrix[i][j] = tmp++;
	  matrix[i * Q + j] = tmp;
	  matrixBuffer.put(i * Q + j, tmp);
	  tmp++;
	}
      }
      System.out.println ("\nmatrix:\n");	/* print matrix		*/
      PrintArray.p2Din1D (P, Q, matrix);
    }
    if (mytid == 0)
    {
      /* send one column to each process				*/
      for (i = 0; i < Q; ++i)
      {
	MPI.COMM_WORLD.send (slice(matrixBuffer, i), 1, column_t,
			     i + 1, 0);
      }
    }
    else
    {
      MPI.COMM_WORLD.recv (column, P, MPI.DOUBLE, 0, 0);
      /* Each process prints its column. The output will probably
       * intermingle on the screen so that you must use
       * "-output-filename" in Open MPI.
       */
      Thread.sleep (SLEEP_FACTOR * mytid);/* sleep to get ordered output*/	
      System.out.println ("\nColumn of process " + mytid + "\n");
      PrintArray.P1D (P, NUM_ELEM_PER_LINE, column);
    }
    column_t.free ();
    MPI.Finalize ();
  }
}
