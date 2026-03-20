# Cluster Usage Guide

## NEVER Run Anything on aurometalsaurus

aurometalsaurus is the login/head node. NEVER run scripts, training, or any computation on it. It is shared by everyone and running jobs on it will slow down the entire cluster for all users. Always use a compute node.

## Access a Compute Node

SSH into a compute node first:

```bash
srun --nodes=1 --nodelist=lenurple --pty /bin/bash -l
srun --nodes=1 --nodelist=sepia --pty /bin/bash -l
srun --nodes=1 --nodelist=byzantium --pty /bin/bash -l
srun --nodes=1 --nodelist=cerulean --pty /bin/bash -l
```

If one node is busy, try another. Or let SLURM pick:

```bash
srun --nodes=1 --pty /bin/bash -l
```

## Create a Conda Environment

```bash
# Create a new environment with a specific Python version
conda create -n myenv python=3.11 -y

# Activate it
conda activate myenv

# Install your packages
uv pip install -r requirements.txt
```

To delete an environment you don't need anymore:

```bash
conda env remove -n myenv
```

## Activate Environment

```bash
conda activate neuro_ezr
```

## Install Packages

Use `uv pip` (faster than regular pip):

```bash
uv pip install package_name
uv pip install -r requirements.txt
```

Fallback to regular pip if uv fails:

```bash
pip install package_name
```

## Run Long Jobs in Background

Use `nohup` so jobs survive if your SSH disconnects:

```bash
nohup python -u script.py > output.log 2>&1 &
```

Check if it's still running:

```bash
ps aux | grep script.py
```

## Kill a Running Process

```bash
# Find the process ID (PID)
ps aux | grep script.py

# Kill it by PID
kill PID

# If it won't die, force kill
kill -9 PID

# Kill all your python processes (careful with this)
pkill -u $USER python
```

## Select GPU

```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 python script.py

# Use GPU 1
CUDA_VISIBLE_DEVICES=1 python script.py

# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 python script.py
```

## Monitoring

```bash
htop          # CPU and memory usage
nvitop        # GPU usage (like htop for GPUs)
nvidia-smi    # GPU status snapshot
```

## Git Basics

```bash
git status                    # See what changed
git add file1.py file2.py     # Stage specific files
git commit -m "message"       # Commit
git push origin main          # Push to GitHub
```

## Troubleshooting

```bash
# Check available nodes
sinfo -N

# List conda environments
conda env list

# Check installed packages
uv pip list

# Exit compute node
exit


# to start cli
npx @google/gemini-cli chat
```
