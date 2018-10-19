/* A  small MPI Java program that creates a vector and prints
 * its size and extent.
 *
 * "mpijavac" and Java-bindings are available in "Open MPI
 * version 1.7.4" or newer.
 *
 *
 * Class file generation:
 *   mpijavac SizeExtentMain.java
 *
 * Usage:
 *   mpiexec [parameters] java [parameters] SizeExtentMain
 *
 * Examples:
 *   mpiexec java SizeExtentMain
 *   mpiexec java -cp $HOME/mpi_classfiles SizeExtentMain
 *
 *
 * File: SizeExtentMain.java		Author: S. Gross
 * Date: 19.05.2014
 *
 */

import mpi.*;

public class SizeExtentMain
{
  static final int COUNT = 2;
  static final int BLOCKLENGTH = 2;
  static final int STRIDE = 4;
  static final int SIZEOF_DOUBLE = 8;

  public static void main (String args[]) throws MPIException
  {
    int  mytid,				/* my task id			*/
	 size;				/* size of vector		*/
    long extent, trueExtent,		/* extent of (resized) vector	*/
	 lb, trueLb;			/* lower bound of (resized) vect*/
    Datatype vector_t,			/* strided vector		*/
	     tmp_vector_t;

    MPI.Init (args);
    mytid  = MPI.COMM_WORLD.getRank ();

    /* Build the new type for a strided vector and resize the extent
     * of the new datatype in such a way that the extent of the whole
     * vector looks like just one element.
     */
    tmp_vector_t = Datatype.createVector (COUNT, BLOCKLENGTH, STRIDE,
					  MPI.DOUBLE);
    vector_t = Datatype.createResized (tmp_vector_t, 0, SIZEOF_DOUBLE);
    vector_t.commit ();
    tmp_vector_t.free ();
    if (mytid == 0)
    {
      size	 = vector_t.getSize ();
      extent	 = vector_t.getExtent ();
      trueExtent = vector_t.getTrueExtent ();
      lb	 = vector_t.getLb ();
      trueLb	 = vector_t.getTrueLb ();
      System.out.println ("strided vector:\n" +
	  "  sizeof (old data type): " + SIZEOF_DOUBLE + "\n" +
	  "  count:                  " + COUNT + "\n" +
	  "  blocklength:            " + BLOCKLENGTH + "\n" +
	  "  stride:                 " + STRIDE + "\n" +
	  "  size:                   " + size + "\n" +
	  "  lower bound:            " + lb + "\n" +
	  "  extent:                 " + extent + "\n" +
	  "  true lower bound:       " + trueLb + "\n" +
	  "  true extent:            " + trueExtent + "\n");
    }
    vector_t.free ();
    MPI.Finalize ();
  }
}
