// must compile with -std=c99 -Wall -o checkdiv 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char *argv[]){
  int size, rank;
  unsigned int x, n;
  unsigned int i; //loop index
  FILE * fp; //for creating the output file
  char filename[100]=""; // the file name
  char * numbers; //the numbers in the range [2, N]

  clock_t start_p1, start_p2, start_p3, end_p1, end_p2, end_p3;

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

    n = (unsigned int)atoi(argv[1]); 
    x = (unsigned int)atoi(argv[2]);

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

  end_p1 = clock();
  //end of part 1
  /////////////////////////////////////////

  printf("process %d: n = %d, x = %d \n", 
        rank,
        n, 
        x);

  /////////////////////////////////////////
  //start of part 2
  // The main computation part starts here
  start_p2 = clock();

  int local_array[n];
  int curr = 0;

  for (int num = 2 + rank; num <= n; num = num + size) {
    printf("process %d: n = %d\n", 
        rank,
        num);
    if (num % x == 0) {
      local_array[curr] = x;
      curr++;
    }
  }
  int received[size];
  int disp[size];
  // MPI_Gather(&local_array, curr, MPI_INT, &local_array, n, MPI_INT, 0, MPI_COMM_WORLD);
  int results[n];

  // MPI_Gatherv(&local_array, curr, MPI_INT, &results, &received, &disp, MPI_INT, 0, MPI_COMM_WORLD);

  end_p2 = clock();
  // end of the main computation part
  //end of part 2
  /////////////////////////////////////////


  /////////////////////////////////////////
  //start of part 3
  // Writing the results in the file

  //forming the filename

  start_p3 = clock();

  for ( i = 0 ; i < size ; i++ )
    {
        received[i] = 0 ;
    }

  if (rank == 0) {
    MPI_Gatherv(local_array, curr, MPI_INT, results, received, disp, MPI_INT, 0, MPI_COMM_WORLD);

    strcpy(filename, argv[1]);
    strcat(filename, ".txt");

    if( !(fp = fopen(filename,"w+t")))
    {
      printf("Cannot create file %s\n", filename);
      exit(1);
    }

    //Write the numbers divisible by x in the file as indicated in the lab description.
    for (int process = 0; process < size; process++) {
      for (i = 0; i < received[process]; i++) { 
        printf("ddd %d\n", disp[process] + i);
        fprintf(fp, "%d \n", results[disp[process] + i]); 
        printf("result %d\n", results[disp[process] + i]);
        
      }
    }


    // for(i=0;i<=received[0] + received[1];i++){ 
    //   fprintf(fp, "%d \n", results[i]); 
    //   printf("result %d\n", 
    //     results[i]);
    // } 

    fclose(fp);
  } else {
    MPI_Gatherv(local_array, curr, MPI_INT, results, received, disp, MPI_INT, 0, MPI_COMM_WORLD);
  }

  end_p3 = clock();
  //end of part 3
  /////////////////////////////////////////

  /* Print  the times of the three parts */
  printf("time of part1 = %lf s part2 = %lf s part3 = %lf s\n", 
        (double)(end_p1-start_p1)/CLOCKS_PER_SEC,
        (double)(end_p2-start_p2)/CLOCKS_PER_SEC, 
        (double)(end_p3-start_p3)/CLOCKS_PER_SEC );

  MPI_Finalize();
  return 0;
}

