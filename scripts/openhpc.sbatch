#!/usr/bin/env bash
#SBATCH --job-name=matmul
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --output=matmul-openhpc-%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Envie o job a partir da raiz do repositório}"
nodes=${SLURM_JOB_NUM_NODES:?Execute com sbatch}
slots=${SLURM_NTASKS:?Defina --ntasks ou --ntasks-per-node em sbatch}
reps=${REPS:-5}
[[ $reps =~ ^[1-9][0-9]*$ ]]
if (( $# == 0 )); then set -- 256 512 1024 2048; fi

export OMP_NUM_THREADS=1 MATMUL_VERIFY=0 LC_ALL=C
build=$(mktemp -d "$PWD/.matmul-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -f "$build/sequential" "$build/mpi"; rmdir "$build"' EXIT
gcc -O3 src/sequential.c -o "$build/sequential"
mpicc -O3 src/mpi.c -o "$build/mpi"

echo "RUN,openhpc,$SLURM_JOB_ID,$nodes,$reps"
echo "Data (UTC): $(date -u +%FT%TZ)"
echo "Versão do código: $(git rev-parse --short HEAD 2>/dev/null || echo arquivo)"
echo "Nós da alocação:"
scontrol show hostnames "$SLURM_JOB_NODELIST"
gcc --version | sed -n '1p'
mpicc -show
if command -v ompi_info >/dev/null; then
    ompi_info --version
elif command -v mpichversion >/dev/null; then
    mpichversion
fi
srun --nodes=1 --ntasks=1 --exact lscpu

mpi_flags=()
if [[ -n ${MPI_TYPE:-} ]]; then mpi_flags+=(--mpi="$MPI_TYPE"); fi

run_mpi() {
    local ranks=$1
    shift
    local output
    output=$(srun --nodes="$nodes" --ntasks="$ranks" --exact --distribution=block:block \
        --cpu-bind=cores --kill-on-bad-exit=1 "${mpi_flags[@]}" "$build/mpi" "$@")
    if ! awk -F, -v ranks="$ranks" -v nodes="$nodes" '
        $1 == "RESULT" { results++; actual_ranks = $6 }
        $1 == "DETAIL" { details++; actual_nodes = $3 }
        END { exit !(results == 1 && details == 1 && actual_ranks == ranks && actual_nodes == nodes) }
    ' <<< "$output"; then
        printf 'Esperados %s ranks em %s nó(s). Confira o tipo MPI do srun.\n%s\n' \
            "$ranks" "$nodes" "$output" >&2
        return 1
    fi
    printf '%s\n' "$output"
}

# Uma dimensão retangular com resto exercita a divisão em todos os nós alocados.
validation=$(MATMUL_VERIFY=1 run_mpi "$slots" "$((slots * 4 + 1))" 23 31)
echo "Validação elemento a elemento com $slots ranks em $nodes nó(s) concluída."
printf '%s\n' "$validation" | sed 's/^/VALIDATION,/'

ranks_to_test=("$slots")
if (( nodes == 1 )); then
    ranks_to_test=()
    for (( ranks = 1; ranks < slots; ranks *= 2 )); do ranks_to_test+=("$ranks"); done
    ranks_to_test+=("$slots")
fi

for n in "$@"; do
    for (( rep = 1; rep <= reps; rep++ )); do
        srun --nodes=1 --ntasks=1 --exact --cpu-bind=cores "$build/sequential" "$n"
        for ranks in "${ranks_to_test[@]}"; do
            run_mpi "$ranks" "$n"
        done
    done
done
echo "DONE,openhpc,$SLURM_JOB_ID"
