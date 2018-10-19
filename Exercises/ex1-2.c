#include <stdio.h>
#include <time.h>


unsigned long long fibonacci_seq(int n) 
{
	switch(n)
	{
		case 0: 
			return 0;
		case 1:
			return 1;
		default:
			;
			unsigned long long fib1 = 1;
			unsigned long long fib2 = 1;
			unsigned long long tmp = 0;
			for(int i = 2; i < n; ++i)
			{
				tmp = fib1 + fib2;
				fib1 = fib2;
				fib2 = tmp;
			}
			return fib2;
			
	}
}

int main(void)
{
	time_t start = time(NULL);
	int n = 100;
	unsigned long long fib = fibonacci_seq(n);
	printf("Completed, Fibonacci #%d is %d, CPU time: %d, Real time: %d\n", n, fib, clock(), start-time(NULL));
}
