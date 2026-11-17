#include "common.h"
#include <mpi.h>

int main(int argc, char **argv) {
    dims_t d = parse_dims(argc, argv);
    if (d.n > INT_MAX / d.m || d.m > INT_MAX / d.p || d.n > INT_MAX / d.p) {
        fprintf(stderr, "Matrizes excedem o limite de contagens int das coletivas MPI\n");
        return EXIT_FAILURE;
    }

    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int base = d.n / ranks, rest = d.n % ranks;
    int rows = base + (rank < rest);
    double *local_A = alloc_matrix(rows, d.m, 0);
    double *local_C = alloc_matrix(rows, d.p, 1);
    double *B = alloc_matrix(d.m, d.p, 0);
    double *A = NULL, *C = NULL;
    int *a_counts = NULL, *a_offsets = NULL, *c_counts = NULL, *c_offsets = NULL;
    if (rank == 0) {
        A = alloc_matrix(d.n, d.m, 0);
        C = alloc_matrix(d.n, d.p, 0);
        init_matrices(d, A, B);

        int *layout = malloc(4 * (size_t)ranks * sizeof(*layout));
        if (!layout) {
            fprintf(stderr, "Falha ao alocar a distribuição MPI\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        a_counts = layout;
        a_offsets = a_counts + ranks;
        c_counts = a_offsets + ranks;
        c_offsets = c_counts + ranks;
        int first = 0;
        for (int r = 0; r < ranks; r++) {
            int count = base + (r < rest);
            a_counts[r] = count * d.m;
            a_offsets[r] = first * d.m;
            c_counts[r] = count * d.p;
            c_offsets[r] = first * d.p;
            first += count;
        }
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int node_rank;
    MPI_Comm_rank(node, &node_rank);
    int leader = (node_rank == 0), nodes = 0;
    MPI_Reduce(&leader, &nodes, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&node);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    MPI_Scatterv(A, a_counts, a_offsets, MPI_DOUBLE,
                 local_A, rows * d.m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    MPI_Bcast(B, d.m * d.p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < d.m; j++) {
            double a_ij = local_A[(size_t)i * d.m + j];
            for (int k = 0; k < d.p; k++)
                local_C[(size_t)i * d.p + k] += a_ij * B[(size_t)j * d.p + k];
        }

    double t3 = MPI_Wtime();
    MPI_Gatherv(local_C, rows * d.p, MPI_DOUBLE,
                C, c_counts, c_offsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t4 = MPI_Wtime();

    double local_compute = t3 - t2, max_compute = 0.0;
    MPI_Reduce(&local_compute, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int valid = 1;
    if (rank == 0) {
        valid = verify_matrix(d, A, B, C);
        if (valid) {
            report("mpi", d, ranks, t4 - t0, checksum(d, C));
            printf("DETAIL,mpi,%d,%.9f,%.9f,%.9f,%.9f\n",
                   nodes, t1 - t0, t2 - t1, max_compute, t4 - t3);
        }
    }
    free(a_counts);
    free(A);
    free(B);
    free(C);
    free(local_A);
    free(local_C);
    MPI_Finalize();
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
