

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
    int num_bins, num_threads, num_nums;
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

    #pragma omp parallel for num_threads(num_threads) reduction(+:histogram)
    for(i = 0; i < num_nums; i++) {
        // We want to map our numbers from [0, 20] -> [0, num_bins]
        histogram[(int)(nums[i] * scaled_bins)]++;
    }

    for(i = 0; i < num_bins; i++) {
        printf("(%lf, %lf) --- ", ((float)i / (float)num_bins * 20.0),  (float)(((float)i + 1) / (float)num_bins * 20.0));
        printf("bin[%d] = %d\n", i, histogram[i]);
    }

    return 0;
}

