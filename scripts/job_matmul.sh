#!/bin/bash
# job_matmul.sh - multiplicação de matrizes com MPI no cluster openhpc (UFSCar).
# Baseado no modelo job_mpi.sh da disciplina.
#
# Coloque na MESMA pasta: este script, common.h, sequential.c e mpi.c.
#
#   bash job_matmul.sh       # compila AQUI (nó de login) e põe na fila
#   squeue -u $USER          # PD = na fila, R = rodando, sumiu da lista = terminou
#   cat saida-<numero>.out
#
# Os nós de computação não têm os cabeçalhos do C, então o script compila no nó
# de login e só depois chama o sbatch sozinho (use 'bash', não 'sbatch').

#SBATCH -J matmul-grupo            # nome que aparece no squeue
#SBATCH -o saida-%j.out            # %j vira o número do job
#SBATCH -p fast,normal,slow        # roda na que liberar primeiro (todas com nós de 40 CPUs)
#SBATCH --exclude=c23              # c23 tem 96 CPUs e os outros 40: misturar desbalanceia
#SBATCH -N 2                       # número de MÁQUINAS (2 = memória distribuída de verdade)
#SBATCH --ntasks-per-node=8        # processos por máquina. Total = N x este valor (máx. 20)
#SBATCH --cpus-per-task=2          # NÃO MUDE: um núcleo físico inteiro (2 CPUs com HT) por processo
#SBATCH -t 00:40:00                # tempo máximo (estimado: 15 a 25 min)
#SBATCH --mem-per-cpu=1G           # rank 0 guarda A, B e C completas (384 MiB com n = 4096)

# ============================ CONFIGURE AQUI ============================
TAMANHOS=${TAMANHOS:-"512 1024 2048 4096"}   # ordens n = m = p medidas
REPETICOES=${REPETICOES:-3}                 # repetições de cada configuração
# =======================================================================

# ------------------------------------------------------------------------
# Fase 1 - nó de login ('bash job_matmul.sh'): confere, compila, põe na fila.
# ------------------------------------------------------------------------
if [ -z "$SLURM_JOB_ID" ]; then
  cd "$(dirname "$0")" || exit 1
  SCRIPT=$(basename "$0")
  for f in common.h sequential.c mpi.c; do
    [ -f "$f" ] || { echo "ERRO: '$f' não está em $PWD"; exit 1; }
  done
  echo "compilando sequential.c e mpi.c ..."
  gcc -O3 -o sequential sequential.c || { echo "ERRO: compilação falhou. Nada foi para a fila."; exit 1; }
  mpicc -O3 -o mpi mpi.c || { echo "ERRO: compilação falhou. Nada foi para a fila."; exit 1; }
  # TAMANHOS e REPETICOES seguem para o job pelo ambiente (sbatch exporta o ambiente).
  export TAMANHOS REPETICOES
  sbatch $SBATCH_EXTRA "$SCRIPT"
  exit $?
fi

# ------------------------------------------------------------------------
# Fase 2 - nó de computação (o Slurm executa esta parte).
# ------------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1

# OBRIGATÓRIO no openhpc: a rede 192.168.22.x bloqueia conexões TCP entre nós.
export OMPI_MCA_btl_tcp_if_include=200.18.104.0/24
export OMPI_MCA_oob_tcp_if_include=200.18.104.0/24
export LC_ALL=C MATMUL_VERIFY=0

for bin in sequential mpi; do
  if [ ! -x "$bin" ] || [ "$bin.c" -nt "$bin" ] || [ common.h -nt "$bin" ]; then
    echo "ERRO: '$bin' não existe ou é mais antigo que o fonte. Envie com 'bash job_matmul.sh'."
    exit 1
  fi
done

NODES=$SLURM_JOB_NUM_NODES
PER_NODE=$SLURM_NTASKS_PER_NODE
TOTAL=$SLURM_NTASKS
echo "RUN,openhpc,$SLURM_JOB_ID,$NODES,$PER_NODE,$REPETICOES"
echo "job=$SLURM_JOB_ID  maquinas=$SLURM_JOB_NODELIST  processos=$TOTAL"
date -u +"Data (UTC): %FT%TZ"
echo "CPU: $(lscpu | sed -n 's/^Model name: *//p')"
gcc --version | head -1
mpirun --version | head -1
echo

echo "=== processos por máquina ==="
mpirun -n "$TOTAL" --map-by node hostname | sort | uniq -c
echo

# packed: os ranks enchem uma máquina antes da outra (até PER_NODE ranks em 1 nó).
# spread: os ranks se alternam entre as máquinas (memória distribuída de verdade).
run_mpi() {
  local ranks=$1 map=$2; shift 2
  local flag=slot
  [ "$map" = spread ] && flag=node
  echo "MAP,$map,$ranks"
  mpirun -n "$ranks" --map-by "$flag" ./mpi "$@"
}

echo "=== validação elemento a elemento (n x 23 x 31, com resto na divisão) ==="
MATMUL_VERIFY=1 run_mpi "$TOTAL" spread $((TOTAL * 4 + 1)) 23 31 || exit 1
MATMUL_VERIFY=1 run_mpi "$PER_NODE" packed $((PER_NODE * 4 + 1)) 23 31 || exit 1
echo

PACKED=()
for (( r = 1; r <= PER_NODE; r *= 2 )); do PACKED+=("$r"); done
[ "${PACKED[-1]}" -ne "$PER_NODE" ] && PACKED+=("$PER_NODE")
SPREAD=()
for (( r = 2; r <= TOTAL; r *= 2 )); do SPREAD+=("$r"); done
[ "${SPREAD[-1]}" -ne "$TOTAL" ] && SPREAD+=("$TOTAL")
echo "Ranks em 1 nó (packed): ${PACKED[*]} | ranks em $NODES nós (spread): ${SPREAD[*]}"

INICIO=$(date +%s)
for n in $TAMANHOS; do
  echo "=== n = $n ==="
  for (( rep = 1; rep <= REPETICOES; rep++ )); do
    ./sequential "$n"
    for r in "${PACKED[@]}"; do run_mpi "$r" packed "$n" || exit 1; done
    for r in "${SPREAD[@]}"; do run_mpi "$r" spread "$n" || exit 1; done
  done
done
echo "DONE,openhpc,$SLURM_JOB_ID,$(( $(date +%s) - INICIO ))"
date
