#include <stdio.h>

#define P 4
#define Q 6
#define R 8


void matrix_mul(int a[P][Q], int b[Q][R], int c[P][R])
{
	for(int i = 0; i < P; ++i)
	{
		for(int j = 0; j < R; ++j)
		{
			// This are independent operations
			c[i][j] = 0;
			for(int k = 0; k < Q; ++k)
			{
				c[i][j] += a[i][k]*b[k][j];
			}
		}
	}
}

void print_matrix(int h, int w, int matrix[h][w])
{
	for(int i = 0; i < P; ++i)
	{
		for(int j = 0; j < Q; ++j)
		{
			printf("%d ", matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(void)
{
	int a[P][Q] = {
			{1,2,3,4,5,6},
			{1,2,3,4,5,6},
			{1,2,3,4,5,6},
			{1,2,3,4,5,6}
		};
	
	int b[Q][R] = {
			{1,2,3,4,5,6,7,8},
			{1,2,3,4,5,6,7,8},
			{1,2,3,4,5,6,7,8},
			{1,2,3,4,5,6,7,8},
			{1,2,3,4,5,6,7,8},
			{1,2,3,4,5,6,7,8}
		};
	int c[P][R];
	matrix_mul(a,b,c);
	print_matrix(P,Q,a);
	print_matrix(Q,R,b);
	print_matrix(P,R,c);
}
