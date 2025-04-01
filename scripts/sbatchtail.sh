out=$(sbatch $1)
sleep 1
eval "tail -f slurm-${out:(-4)}.out"
