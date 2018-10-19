/* A small MPI Java program that prints the sizes of MPI data types.
 *
 * "mpijavac" and Java-bindings are available in "Open MPI
 * version 1.7.4" or newer.
 *
 *
 * Class file generation:
 *   mpijavac MpiDatatypeSizesMain.java
 *
 * Usage:
 *   mpiexec [parameters] java [parameters] MpiDatatypeSizesMain
 *
 * Examples:
 *   mpiexec java MpiDatatypeSizesMain
 *   mpiexec java -cp $HOME/mpi_classfiles MpiDatatypeSizesMain
 *
 *
 * File: MpiDatatypeSizesMain.java	Author: S. Gross
 * Date: 19.07.2018
 *
 */

import mpi.*;

public class MpiDatatypeSizesMain
{
  public static void main (String args[]) throws MPIException
  {
    int mytid,				/* my task id			*/
	sizeMpiChar,			/* size of MPI data types	*/ 
	sizeMpiShort,
	sizeMpiInt,
	sizeMpiLong,
	sizeMpiByte,
	sizeMpiBoolean,
	sizeMpiFloat,
	sizeMpiDouble;

    MPI.Init (args);
    mytid  = MPI.COMM_WORLD.getRank ();
    sizeMpiChar	   = MPI.CHAR.getSize ();
    sizeMpiShort   = MPI.SHORT.getSize ();
    sizeMpiInt	   = MPI.INT.getSize ();
    sizeMpiLong	   = MPI.LONG.getSize ();
    sizeMpiByte	   = MPI.BYTE.getSize ();
    sizeMpiBoolean = MPI.BOOLEAN.getSize ();
    sizeMpiFloat   = MPI.FLOAT.getSize ();
    sizeMpiDouble  = MPI.DOUBLE.getSize ();
    if (mytid == 0)
    {
      System.out.printf ("  size of MPI.CHAR:    %d\n" +
			 "  size of MPI.SHORT:   %d\n" +
			 "  size of MPI.INT:     %d\n" +
			 "  size of MPI.LONG:    %d\n" +
			 "  size of MPI.BYTE:    %d\n" +
			 "  size of MPI.BOOLEAN: %d\n" +
			 "  size of MPI.FLOAT:   %d\n" +
			 "  size of MPI.DOUBLE:  %d\n",
			 sizeMpiChar,
			 sizeMpiShort,
			 sizeMpiInt,
			 sizeMpiLong,
			 sizeMpiByte,
			 sizeMpiBoolean,
			 sizeMpiFloat,
			 sizeMpiDouble);
    }
    MPI.Finalize ();
  }
}
