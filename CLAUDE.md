# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DGX Cloud Benchmarking provides performance recipes for evaluating AI workloads (pretraining, fine-tuning, inference) across GPU architectures (H100, GB200, B200). It uses containerized benchmarks submitted via SLURM.

## Key Commands

### Installation
```bash
./install.sh  # Interactive setup (installs packages, configures SLURM, downloads containers)
```

### Running Benchmarks
```bash
cd $LLMB_INSTALL  # Always run from installation directory

# List available workloads
llmb-run list
llmb-run list -w <workload_name>  # Details for specific workload

# Single job
llmb-run single -w pretrain_nemotron4 -s 340b --dtype fp8 --scale 256

# Bulk submission (from config file)
llmb-run bulk <job_script> -d  # Dry run first
llmb-run bulk <job_script>

# Submit all installed workloads
llmb-run submit-all --max-scale 256 --dryrun
```

Common flags: `-d/--dryrun` (preview), `-v/--verbose`, `-p/--profile`

### Utility Scripts
```bash
./print_env.sh  # Environment diagnostics
common/parse_train_timing.sh  # Extract performance metrics from logs
```

## Architecture

### Core Components

**llmb-run** (`llmb-run/src/llmb_run/`) - Job launcher CLI:
- `main.py` - CLI entry point (list, single, bulk, submit-all commands)
- `job_launcher.py` - Abstract launcher + NeMo/Sbatch implementations
- `task_manager.py` - Parse job configs, generate task objects
- `workload_validator.py` - Validate workloads against metadata
- `config_manager.py` - Load cluster_config.yaml

**installer** (`installer/installer.py`) - Interactive setup tool that configures SLURM, downloads containers, creates cluster_config.yaml

### Workload Structure

Each workload directory (nemotron/, llama3.1/, deepseek_v3/, etc.) contains:
- `metadata.yaml` - Container images, framework, GPU configs, scales, precisions
- `README.md` - Performance tables and documentation
- Launch scripts for job execution

### Configuration Files

- `cluster_config.yaml` - Created by installer; contains SLURM settings, GPU type, installed workloads, environment variables
- `*/metadata.yaml` - Workload definitions with GPU-specific configs under `run.gpu_configs.{h100,gb200,b200}`

### Key Environment Variables

- `LLMB_REPO` - Repository clone location
- `LLMB_INSTALL` - Top-level installation directory for artifacts
- `LLMB_WORKLOAD` - Workload-specific directory (e.g., `$LLMB_INSTALL/workloads/pretrain_nemotron4`)

Results stored at: `$LLMB_WORKLOAD/experiments/`

## Adding a New Workload

1. Create workload directory with `metadata.yaml` following existing structure
2. Define `general` (workload name, type, framework), `container` (images), `setup` (dependencies), `run` (gpu_configs)
3. Add GPU configurations under `run.gpu_configs.{h100,gb200,b200}` with model_size, dtypes, scales
4. Create README.md with performance tables
5. Update root README.md workload tables

## Dependencies

- Python 3.12.x, Bash 4.2+
- SLURM 22.x+ with task/affinity plugin
- Enroot (container management)
- PyYAML, rich, questionary (Python packages)
