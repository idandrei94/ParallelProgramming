/* Matrix multiplication: c = a * b.
 *
 * This program works with any number of processes to multiply the
 * matrices. It creates a new group with "DEF_NUM_WORKER" (<= n)
 * processes to multiply the matrices, if matrix "a" has n rows
 * and the basic group contains too many processes. If you start
 * the program with up to n processes, it behaves in a similar way as
 * "MatMultWith1toNproc2DarrayIn1DarrayMain.java". Every process must
 * know how many rows it has to work on, so that it can allocate enough
 * buffer space for its rows. Therefore the process with rank 0
 * computes the numbers of rows for each process in the array "numRows"
 * and distributes this array with MPI_Broadcast to all processes. Each
 * process can now allocate memory for its row block in "rows_a" and
 * "rows_c". Now one other task must be done before the rows of matrix
 * "a" can be distributed with MPI_Scatterv: The size of each row
 * block and the offset of each row block must be computed und must
 * be stored in the arrays "sr_counts" and "sr_disps". These data
 * structures must be adjusted to the corresponding values of result
 * matrix "c" before you can call MPI_Gatherv to collect the row
 * blocks of the result matrix.
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
 *   mpijavac MatMultWithAnyProc2DarrayIn1DarrayMain.java
 *
 * Usage:
 *   mpiexec [parameters] java [parameters] \
 *	MatMultWithAnyProc2DarrayIn1DarrayMain
 *
 * Examples:
 *   mpiexec -np 4 java MatMultWithAnyProc2DarrayIn1DarrayMain
 *   mpiexec -np 4 java -cp $HOME/mpi_classfiles \
 *	MatMultWithAnyProc2DarrayIn1DarrayMain
 *
 *
 * File: MatMultWithAnyProc2DarrayIn1DarrayMain.java	Author: S. Gross
 * Date: 09.09.2013
 *
 */

import mpi.*;

public class MatMultWithAnyProc2DarrayIn1DarrayMain
{
  static final int P = 4;		/* # of rows			*/
  static final int Q = 6;		/* # of columns / rows		*/
  static final int R = 8;		/* # of columns			*/
  static final int SLEEP_FACTOR = 500;	/* 500 ms to get ordered output	*/
  static final int DEF_NUM_WORKER = P;	/* # of workers, must be <= P	*/

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
    //    double a[][], b[][],
    //	   c[][],
    //	   rows_a[][],
    //	   rows_c[][];
    double a[], b[],			/* matrices to multiply		*/
	   c[],				/* matrix for result		*/
	   rows_a[],			/* row block of matrix "a"	*/
	   rows_c[];			/* row block of matrix "c"	*/
    String processorName;		/* name of local machine	*/
    Group  groupCommWorld,		/* processes in "basic group"	*/
	   groupWorker,			/* processes in new groups	*/
	   groupOther;
    Intracomm COMM_WORKER,		/* communicators for new groups	*/
	       COMM_OTHER;
    int	   group_w_mem[],		/* array of worker members 	*/
	   group_w_ntasks,		/* # of tasks in "groupWorker"	*/
	   group_o_ntasks,		/* # of tasks in "groupOther"	*/
	   group_w_mytid,		/* my task id in "groupWorker"	*/
	   group_o_mytid;		/* my task id in "groupOther"	*/

    MPI.Init (args);
    mytid  = MPI.COMM_WORLD.getRank ();
    ntasks = MPI.COMM_WORLD.getSize ();

    /* Determine the correct number of processes for this program. If
     * there are more than P processes (i.e., more processes than rows
     * in matrix "a") available, we split the "basic group" into two
     * groups. This program uses a group "groupWorker" to do the real
     * work and a group "groupOther" for the remaining processes of the
     * "basic group". The latter have nothing to do and can terminate
     * immediately. If there are less than or equal to P processes
     * available, all processes belong to group "groupWorker" and
     * group "groupOther" is empty. At first we find out which
     * processes belong to the "basic group" so that we can later
     * easily build both groups.
     */
    groupCommWorld = MPI.COMM_WORLD.getGroup ();
    if (ntasks > P)
    {
      /* There are too many processes, so that we must build a new group
       * with "DEF_NUM_WORKER" processes. At first we must check if
       * DEF_NUM_WORKER has a suitable value.
       */
      if (DEF_NUM_WORKER > P)
      {
	if (mytid == 0)
        {
	  System.err.printf ("\nError:\tInternal program error.\n" +
			     "\tConstant DEF_NUM_WORKER has value" +
			     " %d but must\n" +
			     "\tbe lower than or equal to %d. Please" +
			     " change the\n" +
			     "\tsource code and compile the program" +
			     " again.\n\n",
			     DEF_NUM_WORKER, P);
	}
	groupCommWorld.free ();
	MPI.Finalize ();
	System.exit (0);
      }
      group_w_mem = new int [DEF_NUM_WORKER];
      if (mytid == 0)
      {
	System.out.printf ("\nYou have started %d processes but I " +
			   "need at most %d processes.\n", ntasks, P);
	if (DEF_NUM_WORKER > 1)
        {
	  System.out.printf ("I build a new worker group with %d " +
			     "processes. The processes with\n" +
			     "the following ranks in the basic " +
			     "group belong to the new group:\n  ",
			     DEF_NUM_WORKER);
	}
	else
        {
	  System.out.printf ("I build a new worker group with %d " +
			     "process. The process with\n" +
			     "the following rank in the basic " +
			     "group belongs to the new group:\n  ",
			     DEF_NUM_WORKER);
	}
      }
      for (i = 0; i < DEF_NUM_WORKER; ++i)
      {
	/* fetch some ranks from the basic group for the new worker
	 * group, e.g., the last DEF_NUM_WORKER ranks to demonstrate
	 * that a process may have different ranks in different groups
	 */
	group_w_mem[i] = (ntasks - DEF_NUM_WORKER) + i;
	if (mytid == 0)
        {
	  System.out.printf ("%d   ", group_w_mem[i]);
	}
      }
      if (mytid == 0)
      {
	System.out.printf ("\n\n");
      }
      /* Create group "groupWorker"					*/
      groupWorker = groupCommWorld.incl (group_w_mem);
    }
    else
    {
      /* there are at most as many processes as rows in matrix "a",
       * i.e., we can use the "basic group"
       */
      groupWorker = MPI.COMM_WORLD.getGroup ();
    }
    /* Create group "groupOther" which demonstrates only how to use
     * another group operation and which has  nothing to do in this
     * program.
     */
    groupOther = Group.difference (groupCommWorld, groupWorker);

    groupCommWorld.free ();
    /* Create communicators for both groups. The communicator is only
     * defined for all processes of the group and it is undefined
     * (MPI.COMM_NULL) for all other processes.
     */
    COMM_WORKER = MPI.COMM_WORLD.create (groupWorker);
    COMM_OTHER  = MPI.COMM_WORLD.create (groupOther);


    /* =========================================================
     * ======						  ======
     * ======  Supply work for all different groups.	  ======
     * ======						  ======
     * ======						  ======
     * ====== At first you must find out if a process	  ======
     * ====== belongs to a special group. You can use	  ======
     * ====== Rank () for this purpose. It returns	  ======
     * ====== the rank of the calling process in the	  ======
     * ====== specified group or MPI.UNDEFINED if the	  ======
     * ====== calling process is not a member of the	  ======
     * ====== group.					  ======
     * ======						  ======
     * =========================================================
     */


    /* =========================================================
     * ======  This is the group "groupWorker".		  ======
     * =========================================================
     */
    group_w_mytid = groupWorker.getRank ();
    if (group_w_mytid != MPI.UNDEFINED)
    {
      /* Now let's start with the real work				*/
      group_w_ntasks = COMM_WORKER.getSize ();	/* # of processes	*/
      processorName = MPI.getProcessorName ();
      //      a		    = new double[P][Q];
      //      b		    = new double[Q][R];
      //      c		    = new double[P][R];
      a		    = new double[P * Q];
      b		    = new double[Q * R];
      c		    = new double[P * R];

      /* Each process prints a small message. Messages can intermingle
       * on the screen so that you can use "-output-filename" in
       * Open MPI to separate the messages from different processes.
       * Sleep different times and try to get ordered output.
       */
      Thread.sleep (SLEEP_FACTOR * group_w_mytid);
      System.out.printf ("Worker process %d of %d running on %s.\n",
			 group_w_mytid, group_w_ntasks, processorName);
      COMM_WORKER.barrier ();		/* wait for all other processes	*/
      if (group_w_mytid == 0)
      {
	tmp = 1;
	for (i = 0; i < P; ++i)		/* initialize matrix a		*/
        {
	  for (j = 0; j < Q; ++j)
	  {
	    //	    a[i][j] = tmp++;
	    a[i * Q + j] = tmp++;
	  }
	}
	/* print matrix a						*/
	System.out.printf ("\n(%d,%d)-matrix a:\n\n", P, Q);
	PrintArray.p2Din1D (P, Q, a);
	tmp = Q * R;
	for (i = 0; i < Q; ++i)		/* initialize matrix b		*/
        {
	  for (j = 0; j < R; ++j)
	  {
	    //	    b[i][j] = tmp--;
	    b[i * R + j] = tmp--;
	  }
	}
	/* print matrix b						*/
	System.out.printf ("(%d,%d)-matrix b:\n\n", Q, R);
	PrintArray.p2Din1D (Q, R, b);
      }
      /* send matrix "b" to all processes				*/
      COMM_WORKER.bcast (b, Q * R, MPI.DOUBLE, 0);

      /* allocate memory for array containing the number of rows of a
       * row block for each process
       */
      numRows = new int [group_w_ntasks];

      /* Allocate memory for arrays containing the size and
       * displacement of each row block. It's only necessary for the
       * "root" process, but the compiler reports errors about
       * possibly uninitialized variables, if you put the next two
       * statements into the if-clause.
       */
      sr_counts = new int [group_w_ntasks];
      sr_disps  = new int [group_w_ntasks];
      if (group_w_mytid == 0)
      {
	/* compute number of rows in row block for each process		*/
	tmp = P / group_w_ntasks;
	for (i = 0; i < group_w_ntasks; ++i)
	{
	  numRows[i] = tmp;		/* number of rows		*/
	}
	for (i = 0; i < (P % group_w_ntasks); ++i) /* adjust size 	*/
        {
	  numRows[i]++;
	}
	for (i = 0; i < group_w_ntasks; ++i)
        {
	  sr_counts[i] = numRows[i] * Q; /* size of row block		*/
	}
	sr_disps[0] = 0;		 /* start of i-th row block	*/
	for (i = 1; i < group_w_ntasks; ++i)
        {
	  sr_disps[i] = sr_disps[i - 1] + sr_counts[i - 1];
	}
      }
      /* inform all processes about their amount of data, so that
       * they can allocate enough memory for their row blocks
       */
      COMM_WORKER.bcast (numRows, group_w_ntasks, MPI.INT, 0);
      //      rows_a = new double [numRows[group_w_mytid]][Q];
      //      rows_c = new double [numRows[group_w_mytid]][R];
      rows_a = new double [numRows[group_w_mytid] * Q];
      rows_c = new double [numRows[group_w_mytid] * R];
      /* send row block i of "a" to process i				*/
      COMM_WORKER.scatterv (a, sr_counts, sr_disps, MPI.DOUBLE,
			    rows_a, numRows[group_w_mytid] * Q,
			    MPI.DOUBLE, 0);

      /* Compute row block "group_w_mytid" of result matrix "c".	*/
      for (i = 0; i < numRows[group_w_mytid]; ++i)
      {
	for (j = 0; j < R; ++j)
        {
	  //	  rows_c [i][j] = 0.0;
	  rows_c [i * R + j] = 0.0;
	  for (k = 0; k < Q; ++k)
	  {
	    //	    rows_c[i][j] += rows_a[i][k] * b[k][j]
	    rows_c [i * R + j] += rows_a [i * Q + k] * b[k * R + j];
	  }
	}
      }

      if (group_w_mytid == 0)
      {
	/* adjust "sr_counts" and "sr_disps" for matrix "c"		*/
	for (i = 0; i < group_w_ntasks; ++i)
	{
	  sr_counts[i] = numRows[i] * R;     /* size of row block	*/
	}
	for (i = 1; i < group_w_ntasks; ++i) /* start of i-th row block	*/
	{
	  sr_disps[i] = sr_disps[i - 1] + sr_counts[i - 1];
	}
      }
      /* receive row block i of "c" from process i			*/
      COMM_WORKER.gatherv (rows_c, numRows[group_w_mytid] * R,
			   MPI.DOUBLE, c, sr_counts, sr_disps,
			   MPI.DOUBLE, 0);
      if (group_w_mytid == 0)
      {
	/* print matrix c						*/
	System.out.printf ("(%d,%d)-result-matrix c = a * b:\n\n", P, R);
	PrintArray.p2Din1D (P, R, c);
      }
      COMM_WORKER.free ();
    }


    /* =========================================================
     * ======  This is the group "groupOther".		  ======
     * =========================================================
     */
    group_o_mytid = groupOther.getRank ();
    if (group_o_mytid != MPI.UNDEFINED)
    {
      /* Nothing to do (only to demonstrate how to divide work for
       * different groups).
       */
      group_o_ntasks = COMM_OTHER.getSize ();	/* # of processes	*/
      if (group_o_mytid == 0)
      {
	if (group_o_ntasks == 1)
        {
	  System.out.printf ("\nGroup \"groupOther\" contains %d " +
			     "process which has\n" +
			     "nothing to do.\n\n", group_o_ntasks);
	}
	else
        {
	  System.out.printf ("\nGroup \"groupOther\" contains %d " +
			     "processes which have\n" +
			     "nothing to do.\n\n", group_o_ntasks);
	}
      }
      COMM_OTHER.free ();
    }


    /* =========================================================
     * ======  all groups will reach this point		  ======
     * =========================================================
     */
    groupWorker.free ();
    groupOther.free ();
    MPI.Finalize ();
  }
}
