// must compile with -std=c99 -Wall -o checkdiv 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>


int main(int argc, char *argv[]){
int size, rank;
unsigned int x, n;
FILE * fp; //for creating the output file
char filename[100]=""; // the file name

clock_t start_p1, start_p2, start_p3, end_p1, end_p2, end_p3;

size = 1;
rank = 0;

/////////////////////////////////////////
// start of part 1

start_p1 = clock();
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


end_p1 = clock();
//end of part 1
/////////////////////////////////////////


/////////////////////////////////////////
//start of part 2
// The main computation part starts here

start_p2 = clock();

int curr = 0;
int nth_offset = (int)(n / size);
int local_array[nth_offset];
int* array = (int *)malloc( n * stride * size * sizeof(int) );

for (int num = 2 + rank; num <= n; num++) {
    if (num % x == 0) {
        local_array[curr] = x;
        curr++;
    }
}

int received[size];
int disp[size];
int results[n];

end_p2 = clock();
  
// end of the main compuation part
//end of part 2
/////////////////////////////////////////


/////////////////////////////////////////
//start of part 3
// Writing the results in the file


//forming the filename

start_p3 = clock();

strcpy(filename, argv[1]);
strcat(filename, ".txt");

if( !(fp = fopen(filename,"w+t")))
{
  printf("Cannot create file %s\n", filename);
  exit(1);
}

//Write the numbers divisible by x in the file as indicated in the lab description.

fclose(fp);

end_p3 = clock();
//end of part 3
/////////////////////////////////////////

/* Print  the times of the three parts */
printf("time of part1 = %lf s part2 = %lf s part3 = %lf s\n", 
       (double)(end_p1-start_p1)/CLOCKS_PER_SEC,
       (double)(end_p2-start_p2)/CLOCKS_PER_SEC, 
       (double)(end_p3-start_p3)/CLOCKS_PER_SEC );
return 0;
}

