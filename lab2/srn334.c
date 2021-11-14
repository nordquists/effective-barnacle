

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
    int num_bins, num_threads;
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
    num_threads = (unsigned int)atoi(argv[2]);

    omp_set_num_threads(num_threads);

    nums = malloc(10000000 * sizeof(int));
    
    strcpy(filename, argv[2]);
    strcat(filename, ".txt");

    if(!(fp = fopen(filename,"w+t"))) {
        printf("Cannot create file %s\n", filename);
        exit(1);
    }

    n = 0;
    while (fscanf(fp, "%f", &nums[n++]) != EOF);


    int histogram[num_bins];
    scaled_bins = num_bins * 1/20;

    #pragma omp parallel for reduction(+:histogram)
    for(i = 0; i < n; i++) {
        // We want to map our numbers from [0, 20] -> [0, num_bins]
        histogram[(int)(nums[i] * scaled_bins)]++;
    }

    for(i = 0; i < num_bins; i++) {
        printf("(%lf, %lf) ---", (float)(i / num_bins * 20),  (float)((i + 1) / num_bins * 20));
        printf("bin[%d] = %d\n", i, histogram[i]);
    }

    return 0;
}

