#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include "mpi.h"
#include <math.h>
#include <stdlib.h>

#define Tolerance 0.00001
#define TRUE 1
#define FALSE 0

#define N 5000

double ** A, **B;
int rank, size;
MPI_Status Stat;

int initialize (double **A, int n)
{
   int i,j;

   for (j=0;j<n+1;j++){
     A[0][j]=1.0;
   }
   for (i=1;i<n+1;i++){
      A[i][0]=1.0;
      for (j=1;j<n+1;j++) A[i][j]=0.0;
   }

}



void matrixPrint(double **A)
{
    for(int i = 0; i < 20 + 1; i++)
    {
        for(int j = 0; j < 20 + 1; j++)
        {
            printf(" %2.2f ", A[i][j]);
        }
        printf("\n\n");
    }

    printf("\n\n");

}


void parallelSolve(double **A, int n)
{
    
    
    int chunckSize = ceil((float)n/size);
    double message[N + 1], message2[N + 1], message3[N + 1], message4[N + 1];
    int convergence=FALSE;
    double diff, tmp;
    int i,j, iters=0;
    int for_iters;
    MPI_Request reqs[4];
    MPI_Status stats[4];
    int iter_bound = 21;

    int jLower = (rank * chunckSize) + 1, jUpper = ((rank + 1) * chunckSize) + 1;
   for (for_iters = 1; for_iters < iter_bound; for_iters++) 
   { 
        diff = 0.0;
        //No need to recieve any ghost regions for the first iteration
        if(for_iters != 1)
        {
            // Recieve from your left neighbour
            // Only do it if you have a left neighbour
            if(rank != 0)
            {
                MPI_Recv(message, N + 1, MPI_DOUBLE, rank - 1,  0, MPI_COMM_WORLD, &Stat);
                // Now copy the shadow region into it's proper place.

                #pragma omp parallel for ordered schedule(static) num_threads(2)
                for(i = 1; i < n; i++)
                {
                    A[i][jLower - 1] =  A[i][jLower];
                    A[i][jLower] = message[1 + i];
                }
            }

            // Recieve from your right neighbour
            // Only do this if you have a right neighbour
            if(rank != (size - 1))
            {
                MPI_Recv(message2, N + 1, MPI_DOUBLE, rank + 1,  0, MPI_COMM_WORLD, &Stat);
                // Now copy the shadow region into it's proper place.
                #pragma omp parallel for ordered schedule(static) num_threads(2)
                for(i = 1; i < n; i++)
                {
                    A[i][jUpper] =  A[i][jUpper - 1];
                    A[i][jUpper - 1] = message2[1 + i];
                }
            }
        }
         for (i=1;i<n;i++)
         {
           for (j=jLower;j<jUpper;j++)
           {
             tmp = A[i][j];
             A[i][j] = 0.2*(A[i][j] + A[i][j-1] + A[i-1][j] + A[i][j+1] + A[i+1][j]);
             diff += fabs(A[i][j] - tmp);
           }
         }

       

        //No need to send any ghost regions for the last iteration
        if(for_iters != iter_bound - 1)
        {

            // Send to your left neighbour
            // Only do it if you have a left neighbour
            if(rank != 0)
            {               
                // copy the shadow region to the message array.
                #pragma omp parallel for ordered schedule(static) num_threads(2)
                for(i = 1; i < n; i++)
                {
                    message3[1 + i] = A[i][jLower];
                }
                MPI_Isend(message3, N + 1, MPI_DOUBLE, rank - 1,  0, MPI_COMM_WORLD, &reqs[1]);
            }
            // Send to your right neighbour
            // Only do this if you have a left neighbour
            if(rank != (size - 1))
            {
                #pragma omp parallel for ordered schedule(static) num_threads(2)
                for(i = 1; i < n; i++)
                {
                    message4[1 + i] = A[i][jUpper];
                }
                MPI_Isend(message4, N + 1, MPI_DOUBLE, rank + 1,  0, MPI_COMM_WORLD, &reqs[3]);
            }
        }

    }

    double *message5 = (double *)calloc((n + 2), chunckSize * sizeof(double));
    int f = n - 1;
    if(rank == 0)
    {
        for(int target = 1; target < size; target++)
        {
            int lowerBound = target * chunckSize, upperBound = target == (size - 1) ? n : (target + 1) * chunckSize;
            MPI_Recv(message5, n * chunckSize, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &Stat);
            #pragma omp parallel for ordered schedule(static) num_threads(2)
            for(int i = 1; i < n; i++)
            {
                for(int j = lowerBound; j < upperBound; j++)
                { 
                    //printf("%d, %d, %f\n", i, j + chunckSize*rank, message5[i][j]);
                    A[i][j] = message5[i + (j - lowerBound) * f];
                }
            }
        }
    }
    else
    {
        //printf("%d %d\n", n, (size) * chunckSize);
        
        int lowerBound = rank * chunckSize, upperBound = rank == (size - 1) ? n : (rank + 1) * chunckSize;
        //printf("%d %d \n", upperBound, lowerBound);
        #pragma omp parallel for ordered schedule(static) num_threads(2)
        for(int i = 1; i < n; i++)
        {
            for(int j = lowerBound; j < upperBound; j++)
            {
                //printf("%d %d\n", i, j);
                message5[i + (j - lowerBound) * f] = A[i][j];
            }
        }
        MPI_Send(message5, n * chunckSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

void solve(double **A, int n)
{
   int convergence=FALSE;
   double diff, tmp;
   int i,j, iters=0;
   int for_iters;


   for (for_iters=1;for_iters < 21;for_iters++) 
   { 
     diff = 0.0;

     for (i=1;i<n;i++)
     {
       for (j=1;j<n;j++)
       {
         tmp = A[i][j];
         A[i][j] = 0.2*(A[i][j] + A[i][j-1] + A[i-1][j] + A[i][j+1] + A[i+1][j]);
         diff += fabs(A[i][j] - tmp);
       }
     }
     iters++;

     if (diff/((double)N*(double)N) < Tolerance)
    {
       convergence=TRUE;
       printf("Number of iterations needed: %d\n", iters);
       return;
    }

    } /*for*/
}

double validator(double **A, double **B, int n)
{
    double diff = 0, total = 0;
    for(int i = 0; i < n + 2; i++)
    {
        for(int j = 0; j < n + 2; j++)
        {
            if(A[i][j] != B[i][j])
            {
                //printf("%d %d", i, j);
                return 0;
            }
        }    
    }
    return 1;
}

long usecs (void)
{
  struct timeval t;

  gettimeofday(&t,NULL);
  return t.tv_sec*1000000+t.tv_usec;
}


int main(int argc, char * argv[])
{
   int i, provided;
   MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &provided);
   
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   
   double timeSerial, timeParallel, t_start,t_end;

   A = malloc((N+2) * sizeof(double *));
   for (i=0; i<N+2; i++) {
	   A[i] = calloc((N+2), sizeof(double)); 
   }

   B = malloc((N+2) * sizeof(double *));
   for (i=0; i<N+2; i++) {
       B[i] = calloc((N+2), sizeof(double)); 
   }

   if(rank == 0)
   {
        initialize(A, N);

        t_start = MPI_Wtime();
        solve(A, N);
        t_end = MPI_Wtime();


        timeSerial = t_end - t_start;
        printf("Serial computation time = %f\n", timeSerial);
        //matrixPrint(A);
    }
   initialize(B, N);

   MPI_Barrier(MPI_COMM_WORLD);

   t_start = MPI_Wtime();
   parallelSolve(B, N);
   t_end = MPI_Wtime();

   

	if(rank == 0)
    {
        timeParallel = t_end - t_start;
        printf("Parallel computation time = %f\n", timeParallel);    
        printf("Speedup is: [%.2fx]\n", timeSerial/timeParallel);
        //printf("Difference is %.2f\n", validator(A, B, N));
        //matrixPrint(B);
    }

    MPI_Finalize();

}
