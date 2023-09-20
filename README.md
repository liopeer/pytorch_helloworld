# PyTorch HelloWorld
basic pytorch repo to check functionality (for job submission, multi gpu usage, ...)

[Check Here](https://wiki.vision.ee.ethz.ch/itet/gpuclusterslurm)

make sure that `log` folder exists
```bash
sbatch --job-name=NAME --output=log/%j.out --gres=gpu:1 --mem=10G subscript.sh SCRIPT_PARAMS
```

interactive debugging shell
```bash
srun --time 10 --partition=gpu.debug --gres=gpu:1 --pty bash -i
```
not working yet

# Distributed Data Parallel (DDP)
- faster training on several GPUs: **data parallel**
- model too large for single GPU: **model parallel** to split across multiple GPUs
- 1 process per 1 GPU in DDP
- same model parameters & optimizers, but we split the data (`DistributedSampler`)

## Setup Process Groups