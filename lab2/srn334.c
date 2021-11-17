

/*

Read in file sequentially 


*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    FILE * fp;
    int n, i, t;
    float scaled_bins;
    int num_bins, threads, num_nums;
    clock_t start_io, end_io;
    double start_parallel, end_parallel;

    char filename[100]="";
    float* nums;

    if(argc != 4){
        printf("usage:  ./srn334 num_bins num_threads filename\n");
        printf("num_bins: the number of bins\n");
        printf("num_threads: the number of threads\n");
        printf("filename: the filename\n");
        exit(1);
    } 

    num_bins = (unsigned int)atoi(argv[1]); 
    threads = (unsigned int)atoi(argv[2]);

    start_io = clock();

    strcpy(filename, argv[3]);

    if(!(fp = fopen(filename,"r"))) {
        printf("Cannot create file %s\n", filename);
        exit(1);
    }

    fscanf(fp, "%d", &num_nums);

    nums = malloc(num_nums * sizeof(int));
    
    n = 0;
    while (fscanf(fp, "%f", &nums[n++]) != EOF);
    fclose(fp);
    
    end_io = clock();

    int histogram[num_bins];
    for(i = 0; i < num_bins; i++) histogram[i] = 0;
    scaled_bins = (float)num_bins / 20.0;

    start_parallel = omp_get_wtime(); 
   
    // #pragma omp parallel for num_threads(threads) reduction(+:histogram)
    // for(i = 0; i < num_nums; i++) {
    //     // We want to map our numbers from [0, 20] -> [0, num_bins]
    //     // if(nums[i] == 20.0) printf("Exact 20.0 found. \n");
    //     histogram[(int)(nums[i] * scaled_bins)]++;
    // }

    #pragma omp parallel num_threads(threads)
    {
        int local_histogram[threads][num_bins];
        #pragma omp single
        memset(local_histogram, 0, sizeof(local_histogram[0][0]) * threads * num_bins);

        int tid = omp_get_thread_num(); 

        #pragma omp for 
        for(i = 0; i < num_nums; i++) {
            local_histogram[tid][(int)(nums[i] * scaled_bins)]++;
        }

        #pragma omp single
        for(i = 0; i < num_bins; i++) {
            for(t = 0; t < threads; t++) {
                histogram[i] += local_histogram[t][i];
            }
        }
    }

    end_parallel = omp_get_wtime(); 

    for(i = 0; i < num_bins; i++) {
        // printf("(%lf, %lf) --- ", ((float)i / (float)num_bins * 20.0),  (float)(((float)i + 1) / (float)num_bins * 20.0));
        printf("bin[%d] = %d\n", i, histogram[i]);
    }

    // printf("we used %d threads \n", threads);

    printf("time of io %lf s, time of parallel part %lf s\n", 
        ((double)(end_io-start_io)/CLOCKS_PER_SEC),
            (end_parallel-start_parallel));

    return 0;
}

