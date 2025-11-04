#include "stdafx.h"
#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"

int main(int argc, char* argv[]) {
    MPI_INIT(&argc, &argv);

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        char hello[] = "HELLO";
        MPI_Send(hello, _countof(hello), MPI_CHAR, 1, 0, MPI_COMM_WORLD);

    } else if (rank == 1) {
        char hello[12];
        MPI_Recv(hello, _countof(hello), MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        printf("ma  sinucid");
    }

    MPI_Finalize();
    return 0;
}