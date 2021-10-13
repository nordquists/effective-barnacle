// must compile with -std=c99 -Wall -o checkdiv 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>


int main(int argc, char *argv[]){
int i;
int size, rank;
unsigned int x, n;
FILE * fp; //for creating the output file
char filename[100]=""; // the file name
char * numbers;
int* results;


clock_t start_p1, start_p3, end_p1, end_p3;
double start_p2, end_p2;

MPI_Init(&argc, &argv);

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

/////////////////////////////////////////
// start of part 1

start_p1 = clock();

if (rank == 0) {
    // Check that the input from the user is correct.
    if(argc != 3){
        printf("usage:  ./checkdiv N x\n");
        printf("N: the upper bound of the range [2,N]\n");
        printf("x: divisor\n");
        exit(1);
    }  
    printf("ARGS MATCH \n");
    n = (unsigned int)atoi(argv[1]); 
    x = (unsigned int)atoi(argv[2]);
    printf("READ LINE \n");
    // Process 0 must send the x and n to each process.
    // Other processes must, after receiving the variables, calculate their own range.
    for (int dest = 1; dest < size; dest++) {
        MPI_Send(&n, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(&x, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    }
} else {
    MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&x, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
// Process 0 must send the x and n to each process.
// Other processes must, after receiving the variables, calculate their own range.

end_p1 = clock();
//end of part 1
/////////////////////////////////////////


/////////////////////////////////////////
//start of part 2
// The main computation part starts here

start_p2 = MPI_Wtime(); 


printf("START PT 2 \n");
n = n + 1;
int curr = 0;
int remainder = (n - 2) % size; // tells us how many processes must do 1 additional number
int split = (n - 2) / size;
int max_local_array = split / 2 + 1;

int* local_array = malloc(max_local_array * sizeof(int));
// int local_array[max_local_array];
// int results[max_local_array * size];
// printf("split!!!! %d \n", split);
int extra_offset = 2;
int extra = 0;

printf("VALUES INITALIZED \n");

if (rank < remainder) {
    extra = 1;
    extra_offset = extra_offset + rank;
} else {
    extra_offset = extra_offset + remainder;
}

printf("process %d: range=[%d, %d)\n", 
        rank,
        extra_offset + split * rank,
        extra_offset + extra + split * (rank + 1));

for ( i = 0 ; i < max_local_array ; i++ ) {
        local_array[i] = -1 ;
    }

for (int num = extra_offset + split * rank; num < extra_offset + extra + split * (rank + 1); num++) {
    if (num % x == 0) {
        // printf("process %d: FOUND = %d\n", 
        //     rank,
        //     num);
        local_array[curr] = num;
        curr++;
    }
}

end_p2 = MPI_Wtime(); 

double max_p2;
double elapsed_time = end_p2 - start_p2;

MPI_Reduce(&elapsed_time, &max_p2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

// end of the main compuation part
//end of part 2
/////////////////////////////////////////


/////////////////////////////////////////
//start of part 3
// Writing the results in the file

// results = (int *)malloc( (split + 1) * size * sizeof(int) );
//     int i;
//     for ( i = 0 ; i < (split + 1) * size ; i++ ) {
//         results[i] = -1 ;
//     }
  
//     MPI_Gather(local_array, curr, MPI_INT, results, (split + 1) * size, MPI_INT, 0, MPI_COMM_WORLD);

//forming the filename
results = NULL;
start_p3 = clock();
if (rank == 0) {
    results = (int *)malloc( ( max_local_array* size) * sizeof(int) );
    for ( i = 0 ; i < (max_local_array * size) ; i++ ) {
        results[i] = -1 ;
    }
  
    MPI_Gather(local_array, max_local_array, MPI_INT, results, max_local_array, MPI_INT, 0, MPI_COMM_WORLD);

    strcpy(filename, argv[1]);
    strcat(filename, ".txt");

    if( !(fp = fopen(filename,"w+t")))
    {
      printf("Cannot create file %s\n", filename);
      exit(1);
    }

    for(i=0;i < max_local_array * size;i++){ 
      if (results[i] != -1) {
        fprintf(fp, "%d \n", results[i]); 
      }
    } 
    free(results);
    fclose(fp);
  } else {
    MPI_Gather(local_array, max_local_array, MPI_INT, results, max_local_array, MPI_INT, 0, MPI_COMM_WORLD);
  }
free(local_array);
end_p3 = clock();
//end of part 3
/////////////////////////////////////////

/* Print  the times of the three parts */
if(rank == 0) {
printf("time of part1 = %lf s part2 = %lf s part3 = %lf s\n", 
       (double)(end_p1-start_p1)/CLOCKS_PER_SEC,
       (double)(max_p2)/CLOCKS_PER_SEC, 
       (double)(end_p3-start_p3)/CLOCKS_PER_SEC );

}

MPI_Finalize();

return 0;
}

