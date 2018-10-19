#include <stdio.h>
#include <time.h> 
#include <stdlib.h> 

#define ARR_SIZE 50000
#define ARR_SIZE_BIG 100000

int is_sorted(int arr[], size_t length)
{
    for (size_t i = 0; i < length - 1; ++i)
        if (arr[i] > arr[i+1])
            return 0;
    return 1;
}

void init_arr(int arr[], size_t length)
{
    srand(time(0)); 
    for (size_t i = 0; i < length; ++i)
    {
        arr[i] = rand();
    }
}

void bubble_sort(int arr[], size_t length)
{
    int flag = 1;
    while(flag)
    {
        flag = 0;
        for (size_t i = 0; i < length-1; ++i)
        {
            if(arr[i] > arr[i+1])
            {
                flag = 1;
                arr[i] = arr[i] ^ arr[i+1];
                arr[i+1] = arr[i] ^ arr[i+1];
                arr[i] = arr[i] ^ arr[i+1];
            }
        }
    }
}

void select_sort(int arr[], size_t length)
{
    for(size_t i = 0; i < length-1; ++i)
    {
        for (size_t j = i+1; j < length; ++j)
        {
            if(arr[i] > arr[j])
            {
                arr[i] = arr[i] ^ arr[j];
                arr[j] = arr[i] ^ arr[j];
                arr[i] = arr[i] ^ arr[j];
            }
        }
    }
}

void copy_arr(int source[], int dest[], size_t length)
{
    for (size_t i = 0; i < length; ++i)
    {
        dest[i] = source[i];
    }
}

int main(void)
{
    int template[ARR_SIZE];
    init_arr(template, ARR_SIZE);

    int template_big[ARR_SIZE_BIG];
    init_arr(template_big, ARR_SIZE_BIG);

	printf("Algorithm\t\tN value\t\tCPU time\t\tReal time\n");
	printf("-------------------------------------------------------------------------\n");

    int arr[ARR_SIZE];
    int arr_big[ARR_SIZE_BIG];
    
    copy_arr(template, arr, ARR_SIZE);

    time_t start_time = time(NULL);
	clock_t start_clock = clock();
    
    bubble_sort(arr, ARR_SIZE);
    if(!is_sorted(arr, ARR_SIZE))
        printf("### ERROR ###");
    int a = arr[ARR_SIZE/2];
    printf("Bubblesort\t\t%d\t\t%.2f sec\t\t%d sec\n", ARR_SIZE, (clock() - start_clock)/1000.0, time(NULL) - start_time);
    
    copy_arr(template_big, arr_big, ARR_SIZE_BIG);

    start_time = time(NULL);
	start_clock = clock();
    
    bubble_sort(arr_big, ARR_SIZE_BIG);
    if(!is_sorted(arr_big, ARR_SIZE_BIG))
        printf("### ERROR ###");
    a = arr_big[ARR_SIZE_BIG/2];
    printf("Bubblesort\t\t%d\t\t%.2f sec\t\t%d sec\n", ARR_SIZE_BIG, (clock() - start_clock)/1000.0, time(NULL) - start_time);
    
    copy_arr(template, arr, ARR_SIZE);

    start_time = time(NULL);
	start_clock = clock();
    
    select_sort(arr, ARR_SIZE);
    if(!is_sorted(arr, ARR_SIZE))
        printf("### ERROR ###");
    a = arr[ARR_SIZE/2];
    printf("Select sort\t\t%d\t\t%.2f sec\t\t%d sec\n", ARR_SIZE, (clock() - start_clock)/1000.0, time(NULL) - start_time);
    
    copy_arr(template_big, arr_big, ARR_SIZE_BIG);

    start_time = time(NULL);
	start_clock = clock();
    
    select_sort(arr_big, ARR_SIZE_BIG);
    if(!is_sorted(arr_big, ARR_SIZE_BIG))
        printf("### ERROR ###");
    a = arr_big[ARR_SIZE_BIG/2];
    printf("Select sort\t\t%d\t\t%.2f sec\t\t%d sec\n", ARR_SIZE_BIG, (clock() - start_clock)/1000.0, time(NULL) - start_time);

    return 0;
}