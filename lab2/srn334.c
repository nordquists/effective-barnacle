

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
    int n, i;
    float scaled_bins;
    int num_bins, threads, num_nums;
    clock_t start_parallel, end_parallel;
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
    
    int histogram[num_bins];
    for(i = 0; i < num_bins; i++) histogram[i] = 0;
    scaled_bins = (float)num_bins / 20.0;

    start_parallel = clock();

    // #pragma omp parallel num_threads(threads)
    // {
    //     #pragma omp single
    //     start_parallel = clock();

    //     #pragma omp for reduction(+:histogram)
    //     for(i = 0; i < num_nums; i++) {
    //         // We want to map our numbers from [0, 20] -> [0, num_bins]
    //         // if(nums[i] == 20.0) printf("Exact 20.0 found. \n");
    //         histogram[(int)(nums[i] * scaled_bins)]++;
    //     }

    //     #pragma omp single
    //     end_parallel = clock();
    // }

    // #pragma omp parallel for num_threads(threads) reduction(+:histogram)
    // for(i = 0; i < num_nums; i++) {
    //     // We want to map our numbers from [0, 20] -> [0, num_bins]
    //     // if(nums[i] == 20.0) printf("Exact 20.0 found. \n");
    //     histogram[(int)(nums[i] * scaled_bins)]++;
    // }

    #pragma omp parallel for num_threads(10) reduction(+:histogram)
    for(i = 0; i < num_nums; i++) {
        // We want to map our numbers from [0, 20] -> [0, num_bins]
        // if(nums[i] == 20.0) printf("Exact 20.0 found. \n");
        histogram[1]++;
    }

    // #pragma omp parallel for num_threads(threads)
    // for(i = 0; i < num_nums; i++) {
    //     // We want to map our numbers from [0, 20] -> [0, num_bins]
    //     // if(nums[i] == 20.0) printf("Exact 20.0 found. \n");
    //     #pragma omp atomic
    //     histogram[(int)(nums[i] * scaled_bins)]++;
    // }

    end_parallel = clock();


    for(i = 0; i < num_bins; i++) {
        printf("(%lf, %lf) --- ", ((float)i / (float)num_bins * 20.0),  (float)(((float)i + 1) / (float)num_bins * 20.0));
        printf("bin[%d] = %d\n", i, histogram[i]);
    }

    printf("time of parallel part %lf s\n", 
        (double)(end_parallel-start_parallel)/CLOCKS_PER_SEC);
    

    return 0;
}

