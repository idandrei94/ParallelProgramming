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

unsigned long long fibonacci_list[1000];

unsigned long long fibonacci_rec(int n)
{
	switch(n)
	{
		case 0: 
			return 0;
		case 1:
			return 1;
		default:
		;
			unsigned long long fib1, fib2;
			if(fibonacci_list[n-2] == 0) 
			{
				fib1 = fibonacci_rec(n-2);
				fibonacci_list[n-2] = fib1;
			}
			else
				fib1 = fibonacci_list[n-2];
			if(fibonacci_list[n-1] == 0) 
			{
				fib2 = fibonacci_rec(n-1);
				fibonacci_list[n-1] = fib2;
			}
			else
				fib2 = fibonacci_list[n-1];
			unsigned long long res = fib1+fib2;
			fibonacci_list[n] = res;
			return res;
			
	}
}

int main(void)
{
	time_t start_time = time(NULL);
	clock_t start_clock = clock();
	int n = 40;
	unsigned long long fib = fibonacci_seq(n);
	printf("Completed sequenctial, Fibonacci #%d is %d, CPU time: %d, Real time: %d\n", n, fib, clock()-start_clock, time(NULL)-start_time);
	start_time = time(NULL);
	start_clock = clock();
	fib = fibonacci_rec(n);
	printf("Completed recurrence, Fibonacci #%d is %d, CPU time: %d, Real time: %d\n", n, fib, clock()-start_clock, time(NULL)-start_time);
}