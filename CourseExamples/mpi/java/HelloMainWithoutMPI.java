/* A very small Java program which prints a message.
 *
 *
 * Class file generation:
 *   javac HelloMainWithoutMPI.java
 *
 * Usage:
 *   java [parameters] HelloMainWithoutMPI
 *
 *
 * File: HelloMainWithoutMPI.java	Author: S. Gross
 * Date: 09.09.2013
 *
 */

import java.net.*;

public class HelloMainWithoutMPI
{
  public static void main (String args[]) throws Exception
  {
    System.out.println ("Hello from " + InetAddress.getLocalHost());
  }
}
