/* Matrix multiplication: c = a * b.
 *
 * This program uses between one and n processes to multiply the
 * matrices if matrix "a" has n rows. If you start the program with
 * one process it has to compute all n rows of the result matrix alone
 * and if you start the program with n processes each process computes
 * one row of the result matrix. Every process must know how many rows
 * it has to work on, so that it can allocate enough buffer space for
 * its rows. Therefore the process with rank 0 computes the numbers of
 * rows for each process in the array "numRows" and distributes this
 * array with MPI_Broadcast to all processes. Each process can now
 * allocate memory for its row block in "rows_a" and "rows_c". Now one
 * other task must be done before the rows of matrix "a" can be
 * distributed with MPI_Scatterv: The size of each row block and the
 * offset of each row block must be computed und must be stored in
 * the arrays "sr_counts" and "sr_disps". These data structures must
 * be adjusted to the corresponding values of result matrix "c" before
 * you can call MPI_Gatherv to collect the row blocks of the result
 * matrix.
 *
 * At the moment (January 2013) the Java interface of Open MPI
 * for the broadcast, gatherv, and scatterv operations can only
 * handle multi-dimensional arrays, if they are stored in a
 * contiguous memory area. Unfortunately Java stores a
 * 2-dimensional array as array of arrays (one 1-dimensional
 * array for each row and another 1-dimensional array for the
 * object IDs of the rows), i.e., each 1-dimensional array is
 * stored in a contiguous memory area, but the whole matrix
 * isn't stored in a contiguous memory area. Therefore it is
 * necessary to simulates a 2-dimensional array in an
 * 1-dimensional array and to compute all indices manually at
 * the moment.
 *
 * "mpijavac" and Java-bindings are available in "Open MPI
 * version 1.7.4" or newer.
 *
 *
 * Class file generation:
 *   mpijavac MatMultWith1toNproc2DarrayIn1DarrayMain.java
 *
 * Usage:
 *   mpiexec [parameters] java [parameters] \
 *	MatMultWith1toNproc2DarrayIn1DarrayMain
 *
 * Examples:
 *   mpiexec -np 4 java MatMultWith1toNproc2DarrayIn1DarrayMain
 *   mpiexec -np 4 java -cp $HOME/mpi_classfiles \
 *	MatMultWith1toNproc2DarrayIn1DarrayMain
 *
 *
 * File: MatMultWith1toNproc2DarrayIn1DarrayMain.java	Author: S. Gross
 * Date: 09.09.2013
 *
 */

import mpi.*;

public class MatMultWith1toNproc2DarrayIn1DarrayMain
{
  static final int P = 4;		/* # of rows			*/
  static final int Q = 6;		/* # of columns / rows		*/
  static final int R = 8;		/* # of columns			*/
  static final int SLEEP_FACTOR = 500;	/* 500 ms to get ordered output	*/

  public static void main (String args[]) throws MPIException,
						 InterruptedException
  {
    int    ntasks,			/* number of parallel tasks	*/
	   mytid,			/* my task id			*/
	   i, j, k,			/* loop variables		*/
	   numRows[],			/* # of rows in a row block	*/
	   sr_counts[],			/* send/receive counts		*/
	   sr_disps[],			/* send/receive displacements	*/
	   tmp;				/* temporary value		*/
    //    double a[][] = new double[P][Q],
    //	   b[][] = new double[Q][R,
    //	   c[][] = new double[P][R],
    //	   rows_a[][],
    //	   rows_c[][];
    double a[] = new double[P * Q],	/* matrices to multiply		*/
	   b[] = new double[Q * R],
	   c[] = new double[P * R],	/* matrix for result		*/
	   rows_a[],			/* row block of matrix "a"	*/
	   rows_c[];			/* row block of matrix "c"	*/
    String processorName;		/* name of local machine	*/

    MPI.Init (args);
    processorName = MPI.getProcessorName ();
    mytid	  = MPI.COMM_WORLD.getRank ();
    ntasks	  = MPI.COMM_WORLD.getSize ();

    if (ntasks > P)
    {
      /* wrong number of processes					*/
      if (mytid == 0)
      {
	String className =
	  new MatMultWith1toNproc2DarrayIn1DarrayMain().getClass().getName();
	System.err.printf ("\nI can use at most %d processes.\n" +
			   "Usage:\n" +
			   "  mpiexec -np n java %s\n" +
			   "where \"n\" is a value between 1 and %d.\n\n",
			   P, className, P);
      }
      MPI.Finalize ();
      System.exit (0);
    }
    /* Each process prints a small message. Messages can intermingle
     * on the screen so that you can use "-output-filename" in
     * Open MPI to separate the messages from different processes.
     * Sleep different times and try to get ordered output.
     */
    Thread.sleep (SLEEP_FACTOR * mytid);
    System.out.printf ("Process %d of %d running on %s.\n",
		       mytid, ntasks, processorName);
    MPI.COMM_WORLD.barrier ();		/* wait for all other processes	*/
    if (mytid == 0)
    {
      tmp = 1;
      for (i = 0; i < P; ++i)		/* initialize matrix a		*/
      {
	for (j = 0; j < Q; ++j)
	{
	  //	  a[i][j] = tmp++;
	  a[i * Q + j] = tmp++;
	}
      }
      /* print matrix a							*/
      System.out.printf ("\n(%d,%d)-matrix a:\n\n", P, Q);
      PrintArray.p2Din1D (P, Q, a);
      tmp = Q * R;
      for (i = 0; i < Q; ++i)		/* initialize matrix b		*/
      {
	for (j = 0; j < R; ++j)
	{
	  //	  b[i][j] = tmp--;
	  b[i * R + j] = tmp--;
	}
      }
      /* print matrix b							*/
      System.out.printf ("(%d,%d)-matrix b:\n\n", Q, R);
      PrintArray.p2Din1D (Q, R, b);
    }
    /* send matrix "b" to all processes					*/
    MPI.COMM_WORLD.bcast (b, Q * R, MPI.DOUBLE, 0);

    /* allocate memory for array containing the number of rows of a
     * row block for each process
     */
    numRows = new int [ntasks];

    /* Allocate memory for arrays containing the size and
     * displacement of each row block. It's only necessary for the
     * "root" process, but the compiler reports errors about
     * possibly uninitialized variables, if you put the next two
     * statements into the if-clause.
     */
    sr_counts = new int [ntasks];
    sr_disps  = new int [ntasks];
    if (mytid == 0)
    {
      /* compute number of rows in row block for each process		*/
      tmp = P / ntasks;
      for (i = 0; i < ntasks; ++i)
      {
	numRows[i] = tmp;		/* number of rows		*/
      }
      for (i = 0; i < (P % ntasks); ++i) /* adjust size		 	*/
      {
	numRows[i]++;
      }
      for (i = 0; i < ntasks; ++i)
      {
	sr_counts[i] = numRows[i] * Q;	/* size of row block		*/
      }
      sr_disps[0] = 0;			/* start of i-th row block	*/
      for (i = 1; i < ntasks; ++i)
      {
	sr_disps[i] = sr_disps[i - 1] + sr_counts[i - 1];
      }
    }
    /* inform all processes about their amount of data, so that
     * they can allocate enough memory for their row blocks
     */
    MPI.COMM_WORLD.bcast (numRows, ntasks, MPI.INT, 0);
    //    rows_a = new double [numRows[mytid]][Q];
    //    rows_c = new double [numRows[mytid]][R];
    rows_a = new double [numRows[mytid] * Q];
    rows_c = new double [numRows[mytid] * R];
    /* send row block i of "a" to process i				*/
    MPI.COMM_WORLD.scatterv (a, sr_counts, sr_disps, MPI.DOUBLE,
			     rows_a, numRows[mytid] * Q, MPI.DOUBLE, 0);

    /* Compute row block "mytid" of result matrix "c".			*/
    for (i = 0; i < numRows[mytid]; ++i)
    {
      for (j = 0; j < R; ++j)
      {
	//	rows_c [i][j] = 0.0;
	rows_c [i * R + j] = 0.0;
	for (k = 0; k < Q; ++k)
        {
	  //	  rows_c[i][j] += rows_a[i][k] * b[k][j]
	  rows_c [i * R + j] += rows_a [i * Q + k] * b[k * R + j];
	}
      }
    }

    if (mytid == 0)
    {
      /* adjust "sr_counts" and "sr_disps" for matrix "c"		*/
      for (i = 0; i < ntasks; ++i)
      {
	sr_counts[i] = numRows[i] * R;	/* size of row block		*/
      }
      for (i = 1; i < ntasks; ++i)	/* start of i-th row block	*/
      {
	sr_disps[i] = sr_disps[i - 1] + sr_counts[i - 1];
      }
    }
    /* receive row block i of "c" from process i			*/
    MPI.COMM_WORLD.gatherv (rows_c, numRows[mytid] * R, MPI.DOUBLE,
			    c, sr_counts, sr_disps, MPI.DOUBLE, 0);
    if (mytid == 0)
    {
      /* print matrix c							*/
      System.out.printf ("(%d,%d)-result-matrix c = a * b:\n\n", P, R);
      PrintArray.p2Din1D (P, R, c);
    }
    MPI.Finalize ();
  }
}
