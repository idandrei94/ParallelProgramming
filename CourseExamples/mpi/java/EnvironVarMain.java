/* A very small Java program which prints the values of some
 * environment variables.
 *
 *
 * Class file generation:
 *   javac EnvironVarMain.java
 *
 * Usage:
 *   java [parameters] EnvironVarMain
 *
 *
 * File: EnvironVarMain.java		Author: S. Gross
 * Date: 09.09.2013
 *
 */

import java.net.*;

public class EnvironVarMain
{
  public static void main (String args[]) throws Exception
  {
    System.out.println ("\nOperating system: " +
			System.getenv ("SYSTEM_ENV") + "    " +
			"Processor architecture: " +
			System.getenv ("MACHINE_ENV") +
			"\n\n  CLASSPATH: " +
			System.getenv ("CLASSPATH") +
			"\n\n  LD_LIBRARY_PATH: " +
			System.getenv ("LD_LIBRARY_PATH"));
  }
}
