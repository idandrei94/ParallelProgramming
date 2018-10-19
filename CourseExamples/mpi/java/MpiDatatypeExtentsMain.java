/* A small MPI Java program that prints the extents of MPI data types.
 *
 * "mpijavac" and Java-bindings are available in "Open MPI
 * version 1.7.4" or newer.
 *
 *
 * Class file generation:
 *   mpijavac MpiDatatypeExtentsMain.java
 *
 * Usage:
 *   mpiexec [parameters] java [parameters] MpiDatatypeExtentsMain
 *
 * Examples:
 *   mpiexec java MpiDatatypeExtentsMain
 *   mpiexec java -cp $HOME/mpi_classfiles MpiDatatypeExtentsMain
 *
 *
 * File: MpiDatatypeExtentsMain.java	Author: S. Gross
 * Date: 19.07.2018
 *
 */

import mpi.*;

public class MpiDatatypeExtentsMain
{
  public static void main (String args[]) throws MPIException
  {
    int mytid,				/* my task id			*/
	extentMpiChar,			/* extent of MPI data types	*/ 
	extentMpiShort,
	extentMpiInt,
	extentMpiLong,
	extentMpiByte,
	extentMpiBoolean,
	extentMpiFloat,
	extentMpiDouble;

    MPI.Init (args);
    mytid  = MPI.COMM_WORLD.getRank ();
    extentMpiChar    = MPI.CHAR.getExtent ();
    extentMpiShort   = MPI.SHORT.getExtent ();
    extentMpiInt     = MPI.INT.getExtent ();
    extentMpiLong    = MPI.LONG.getExtent ();
    extentMpiByte    = MPI.BYTE.getExtent ();
    extentMpiBoolean = MPI.BOOLEAN.getExtent ();
    extentMpiFloat   = MPI.FLOAT.getExtent ();
    extentMpiDouble  = MPI.DOUBLE.getExtent ();
    if (mytid == 0)
    {
      System.out.printf ("  extent of MPI.CHAR:    %d\n" +
			 "  extent of MPI.SHORT:   %d\n" +
			 "  extent of MPI.INT:     %d\n" +
			 "  extent of MPI.LONG:    %d\n" +
			 "  extent of MPI.BYTE:    %d\n" +
			 "  extent of MPI.BOOLEAN: %d\n" +
			 "  extent of MPI.FLOAT:   %d\n" +
			 "  extent of MPI.DOUBLE:  %d\n",
			 extentMpiChar,
			 extentMpiShort,
			 extentMpiInt,
			 extentMpiLong,
			 extentMpiByte,
			 extentMpiBoolean,
			 extentMpiFloat,
			 extentMpiDouble);
    }
    MPI.Finalize ();
  }
}
