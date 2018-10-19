#include <stdio.h>
#include <time.h>

#define CACHE_SIZE 10000

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

unsigned long long fibonacci_list[CACHE_SIZE];

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
			if(n-2 >= CACHE_SIZE || fibonacci_list[n-2] == 0) 
			{
				fib1 = fibonacci_rec(n-2);
				if(n-2 < CACHE_SIZE)
					fibonacci_list[n-2] = fib1;
			}
			else
				fib1 = fibonacci_list[n-2];
			if(n-1 >= CACHE_SIZE || fibonacci_list[n-1] == 0) 
			{
				fib2 = fibonacci_rec(n-1);
				if(n-1 < CACHE_SIZE)
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
	int n = 46;
	unsigned long long fib;
	printf("Algorithm\t\tN value\t\tResult\t\t\tCPU time\tReal time\n\n");
	printf("-----------------------------------------------------------------------------------------\n");
	time_t start_time = time(NULL);
	clock_t start_clock = clock();
	fib = fibonacci_seq(n);++fib;--fib;
	printf("Sequential\t\t%d\t\t%d\t\t%d\t\t%d\n", n, fib, clock()-start_clock, time(NULL)-start_time);
	start_time = time(NULL);
	start_clock = clock();
	fib = fibonacci_rec(n);++fib;--fib;
	printf("Recurrent\t\t%d\t\t%d\t\t%d\t\t%d\n", n, fib, clock()-start_clock, time(NULL)-start_time);
}