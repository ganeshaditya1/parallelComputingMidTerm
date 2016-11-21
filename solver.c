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

double ** A, **B, **C;
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

void parallelSolve(double **A, int n)
{
    
    double start = MPI_Wtime();
    int chunckSize = ceil((float)n/size);
    double message[N + 1], message2[N + 1], message3[N + 1], message4[N + 1];
  
    /*
    
    */
    int convergence=FALSE;
    double diff, tmp;
    int i,j, iters=0;
    int for_iters;
    MPI_Request reqs[4];
    MPI_Status stats[4];

    int jLower = rank * chunckSize + 1, jUpper = rank * chunckSize + 1;
    
   for (for_iters = 1; for_iters < 21; for_iters++) 
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

                #pragma omp parallel for ordered schedule(static) num_threads(4)
                for(i = 1; i < n; i++)
                {
                    A[i][jLower] = message[1 + i];
                }
            }

            // Recieve from your right neighbour
            // Only do this if you have a right neighbour
            if(rank != (size - 1))
            {
                MPI_Recv(message2, N + 1, MPI_DOUBLE, rank + 1,  0, MPI_COMM_WORLD, &Stat);
                // Now copy the shadow region into it's proper place.
                #pragma omp parallel for ordered schedule(static) num_threads(4)
                for(i = 0; i < n; i++)
                {
                    A[i][jUpper] = message2[1 + i];
                }
            }
        }
        
        /*#pragma omp parallel for ordered schedule(dynamic) num_threads(4)
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
           convergence=TRUE;

        }*/

        //No need to send any ghost regions for the last iteration
        if(for_iters != 20)
        {

            // Send to your left neighbour
            // Only do it if you have a left neighbour
            if(rank != 0)
            {               
                // copy the shadow region to the message array.
                #pragma omp parallel for ordered schedule(static) num_threads(4)
                for(i = 0; i < n; i++)
                {
                    message3[1 + i] = 2 * i;
                }
                MPI_Isend(message3, N + 1, MPI_DOUBLE, rank - 1,  0, MPI_COMM_WORLD, &reqs[1]);
            }
            // Send to your right neighbour
            // Only do this if you have a left neighbour
            if(rank != (size - 1))
            {
                #pragma omp parallel for ordered schedule(static) num_threads(4)
                for(i = 0; i < n; i++)
                {
                    message4[1 + i] = i;
                }
                MPI_Isend(message4, N + 1, MPI_DOUBLE, rank + 1,  0, MPI_COMM_WORLD, &reqs[3]);
            }
        }

    }
    

printf("%.2f time elasped\n", MPI_Wtime() - start);
    /* No need to do this anymore
    if(rank != 0 && rank != (size - 1))
    {
        MPI_Waitall(4, reqs, stats);
    }
    else if(rank != 0)
    {
        MPI_Waitall(2, reqs, stats);
    }
    else
    {
        MPI_Waitall(2, &reqs[2], stats);
    }
    */

    
    double *message5 = (double *)malloc(n * chunckSize * sizeof(double));
    int f = n - 1;
    if(rank == 0)
    {
        for(int target = 1; target < size; target++)
        {
            int lowerBound = target * chunckSize, upperBound = target == (size - 1) ? n : (target + 1) * chunckSize;
            MPI_Recv(message5, n * chunckSize, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &Stat);
            #pragma omp parallel for ordered schedule(static) num_threads(4)
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
        #pragma omp parallel for ordered schedule(static) num_threads(4)
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


   for (for_iters=1;for_iters<21;for_iters++) 
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
       convergence=TRUE;

    } /*for*/
}

int validator(double **A, double **B, int n)
{
    for(int i = 0; i < n + 2; i++)
    {
        for(int j = 0; j < n + 2; j++)
        {
            if(A[i][j] != B[i][j])
            {
                printf("%d %d\n", i, j);
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
	   A[i] = malloc((N+2) * sizeof(double)); 
   }

   B = malloc((N+2) * sizeof(double *));
   for (i=0; i<N+2; i++) {
       B[i] = malloc((N+2) * sizeof(double)); 
   }

   C = malloc((N+2) * sizeof(double *));
   for (i=0; i<N+2; i++) {
       C[i] = malloc((N+2) * sizeof(double)); 
   }
   if(rank == 0)
   {
        initialize(A, N);

        t_start = MPI_Wtime();
        //solve(A, N);
        t_end = MPI_Wtime();


        timeSerial = t_end - t_start;
        printf("Serial computation time = %f\n", timeSerial);
    }
   initialize(B, N);
   initialize(C, N);

   t_start = MPI_Wtime();
   parallelSolve(B, N);
   t_end = MPI_Wtime();

   if(rank == 0 && !validator(A, B, N))
   {
        fprintf(stderr, "Outputs don't match!\n");
   }

	if(rank == 0)
    {
        timeParallel = t_end - t_start;
        printf("Parallel computation time = %f\n", timeParallel);    
        printf("Speedup is: [%.2fx]\n", timeSerial/timeParallel);
    }

}
