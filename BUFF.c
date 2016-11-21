
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