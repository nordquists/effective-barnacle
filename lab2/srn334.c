#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    FILE * fp;
    int n, i;
    float scaled_bins;
    int num_bins, threads, num_nums;
    clock_t start_io, end_io;
    double start_parallel, end_parallel;
    char filename[100]="";
    float* nums;


    // START: Reading in arguments
    if(argc != 4){
        printf("usage:  ./srn334 num_bins num_threads filename\n");
        printf("num_bins: the number of bins\n");
        printf("num_threads: the number of threads\n");
        printf("filename: the filename\n");
        exit(1);
    } 

    num_bins = (unsigned int)atoi(argv[1]); 
    threads = (unsigned int)atoi(argv[2]);
    // End: Reading in arguments

    // START: IO Portion 
    start_io = clock();

    strcpy(filename, argv[3]);

    if(!(fp = fopen(filename,"r"))) {
        printf("Cannot create file %s\n", filename);
        exit(1);
    }
    
    // Read the 'number of numbers' in first
    fscanf(fp, "%d", &num_nums);

    nums = malloc(num_nums * sizeof(int));
    
    // Read all of the floating point numbers in
    n = 0;
    while (fscanf(fp, "%f", &nums[n++]) != EOF);
    fclose(fp);
    
    end_io = clock();
    // END: IO Portion 

    // START: Initializing histogram data structure and helper variable
    int histogram[num_bins];
    for(i = 0; i < num_bins; i++) histogram[i] = 0;
    scaled_bins = (float)num_bins / 20.0; // We use this to help us map our numbers to the number of bins
    // End: Initializing histogram data structure and helper variable

    // START: Parallel Portion
    start_parallel = omp_get_wtime(); 
   
    #pragma omp parallel for num_threads(threads) reduction(+:histogram)
    for(i = 0; i < num_nums; i++) {
        // We want to map our numbers from [0, 20) -> [0, num_bins)
        histogram[(int)(nums[i] * scaled_bins)]++;
    }

    end_parallel = omp_get_wtime(); 
    // END: Parallel Portion

    // START: Printing output
    for(i = 0; i < num_bins; i++) {
        printf("bin[%d] = %d\n", i, histogram[i]);
    }
    // End: Printing output

    // START: Printing run times 
    printf("time of io %lf s, time of parallel part %lf s\n", 
        ((double)(end_io-start_io)/CLOCKS_PER_SEC),
            (end_parallel-start_parallel));
    // END: Printing run times 

    return 0;
}

