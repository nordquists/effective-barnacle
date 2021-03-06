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

  /////////////////////////////////////////
  //start of part 2
  // The main computation part starts here
  start_p2 = clock();

  
  // int nth_offset = ((n - 2) / size);
  // int local_array[nth_offset];
  // int curr = 0;


  // for (int num = 2 + rank; num <= n; num = num + size) {
  //   printf("process %d: n = %d\n", 
  //       rank,
  //       num);
  //   if (num % x == 0) {
  //     local_array[curr] = num;
  //     curr++;
  //   }
  // }

  // printf("process %d: range=[%d, %d]\n", 
  //       rank,
  //       2 + nth_offset * rank,
  //       2 + nth_offset * (rank + 1));

  // for (int num = 2 + nth_offset * rank; num < 2 + nth_offset * (rank + 1); num++) {
  //   if (num % x == 0) {
  //     printf("process %d: FOUND = %d\n", 
  //       rank,
  //       num);
  //       local_array[curr] = num;
  //       curr++;
  //   } else {
  //     printf("process %d: i = %d\n", 
  //       rank,
  //       num);
  //   }
  //   }

  int remainder = (n - 2) % size;
  int split = (n - 2) / size;
  int extra_so_far = rank

  int local_array[split + 1];
  int curr = 0;

  printf("process %d: range=[%d, %d)\n", 
        rank,
        2 + split * rank,
        2 + split * (rank + 1));

  for (int num = 2 + ; num < 2 + split + remainder; num++){
    if (num % x == 0) {
      printf("process %d: FOUND = %d\n", 
        rank,
        num);
      local_array[curr] = num;
      curr++;
    
    } else {
      printf("process %d: i = %d\n", 
        rank,
        num);
    }
  }


  end_p2 = clock();
  // end of the main computation part
  //end of part 2
  /////////////////////////////////////////


  /////////////////////////////////////////
  //start of part 3
  // Writing the results in the file

  //forming the filename

  // int* disp = (int *)malloc( size * sizeof(int) );
  // int* received = (int *)malloc( size * sizeof(int) );
  // int* results = (int *)malloc( n * sizeof(int) );

  start_p3 = clock();
  // printf("curr %d\n", curr);
  // for ( i = 0 ; i < n ; i++ )
  //   {
  //       results[i] = -1 ;
  //   }
  
  // // MPI_Gather(local_array, curr, MPI_INT, results, n, MPI_INT, 0, MPI_COMM_WORLD);

  // if (rank == 0) {
  //   strcpy(filename, argv[1]);
  //   strcat(filename, ".txt");

  //   if( !(fp = fopen(filename,"w+t")))
  //   {
  //     printf("Cannot create file %s\n", filename);
  //     exit(1);
  //   }

  //   //Write the numbers divisible by x in the file as indicated in the lab description.
  //   // for (int process = 0; process < size; process++) {
  //   //   printf("aaa %d\n", received[process]);
  //   //   for (i = 0; i < received[process]; i++) { 
  //   //     printf("ddd %d\n", disp[process] + i);
  //   //     fprintf(fp, "%d \n", results[disp[process] + i]); 
  //   //     printf("result %d\n", results[disp[process] + i]);
        
  //   //   }
  //   // }



  //   for(i=0;i<=n;i++){ 
  //     if (results[i] != -1) {
  //       fprintf(fp, "%d \n", results[i]); 
  //       printf("result %d\n", results[i]);
  //     }
  //   } 

  //   fclose(fp);
  // } 
  // free(disp);
  // free(received);
  // free(results);

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

