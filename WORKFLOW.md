# PINN-Experiments: Vast.ai GPU Workflow

This document describes the complete workflow for using Vast.ai GPU rentals with the PINN-Experiments monorepo. It covers instance lifecycle, backup procedures, artifact preservation, folder structure, and git integration.

---

## Table of Contents

1. [Philosophy: Pay Only for What You Use](#philosophy-pay-only-for-what-you-use)
2. [Source of Truth Hierarchy](#source-of-truth-hierarchy)
3. [Folder Structure](#folder-structure)
4. [Project Categories](#project-categories)
5. [Git Integration](#git-integration)
6. [The Two Backup Types](#the-two-backup-types)
7. [Artifact Preservation Policy](#artifact-preservation-policy)
8. [Instance Lifecycle](#instance-lifecycle)
9. [Daily Workflow](#daily-workflow)
10. [Script Reference](#script-reference)
11. [Long-Running Job Safety](#long-running-job-safety)
12. [Pre-Save Checklist](#pre-save-checklist)
13. [Docker Image Hygiene](#docker-image-hygiene)
14. [Emergency Procedures](#emergency-procedures)
15. [Restore Verification](#restore-verification)
16. [Recovery of Lost Artifacts](#recovery-of-lost-artifacts)
17. [Storage Management](#storage-management)
18. [Best Practices](#best-practices)

---

## Prerequisites

Before using this workflow, complete these one-time setup steps:

### 1. Install Vast.ai CLI

```bash
pip install vastai
```

### 2. Configure API Keys

Run the interactive configuration script:

```bash
cd ~/Dev/PINN-Experiments
./configure-keys.sh
```

This will:
- Prompt for your Vast.ai API key (required)
- Optionally configure HuggingFace and W&B tokens
- Generate SSH keys for Vast.ai access
- Verify authentication works

### 3. Verify Setup

```bash
# Check Vast.ai authentication
vastai show user

# List available projects
./vastai-manage.sh list-projects

# View project dashboard
./vastai-manage.sh dashboard
```

### 4. Local Requirements

- **macOS** with standard Unix tools (ssh, scp, tar, gzip)
- **100-300GB free disk space** per active project for backups
- **Docker**: NOT required locally (runs inside Vast.ai instances)

---

## Philosophy: Pay Only for What You Use

Vast.ai charges by the hour. Leaving instances running 24/7 is expensive. This workflow minimizes costs by:

1. **Creating instances only when actively working**
2. **Saving complete backups before destroying instances**
3. **Restoring the exact environment when resuming work**

This approach can reduce costs by 90% or more compared to continuous rental.

**The golden rule**: Never destroy an instance without saving both backups first.

---

## Source of Truth Hierarchy

This workflow defines a strict priority order for what is "authoritative":

| Priority | Source | What It Contains |
|----------|--------|------------------|
| 1 | **Git repository** | Canonical source for all code, configs, docs |
| 2 | **Docker image backups** | Canonical runtime environment snapshot (fast restore) |
| 3 | **File backups** | Canonical workspace snapshot (data/results redundancy) |
| 4 | **Running instance** | Ephemeral and **never** authoritative |

**Rule**: The instance is disposable. If it matters, it must exist in (1), (2), or (3).

This hierarchy prevents panic when instances crash or are destroyed. Nothing on the instance is precious—it's just a temporary compute environment. All valuable work flows into the three permanent storage layers.

---

## Folder Structure

### Repository Layout (Local Mac)

```
~/Dev/PINN-Experiments/                        # Git repo root
├── .git/
├── .gitignore                                 # Ignores docker-backups/ and runtime artifacts
├── README.md                                  # Global overview
├── WORKFLOW.md                                # This file
├── vastai-manage.sh                           # Instance management script
│
├── research-docs/                             # Centralized research documentation
│   ├── ChatGPT-*.md.txt                       # Research plans and literature reviews
│   ├── *_scan.txt                             # Scanned/summarized docs
│   └── parts1-originals/                      # Original exported conversations
│
└── projects/                                  # All projects organized by type
    │
    ├── [PINN Projects]
    │   ├── pinn-lite-foil__project-space/
    │   ├── maxwell-pinn-nif__project-space/
    │   ├── GeoPINN-Manifold__project-space/
    │   ├── cell-path-pinns__project-space/
    │   ├── surfpinn__project-space/
    │   ├── wavepinn-nif__project-space/
    │   └── WavePINN-NIF-ComplexMedia__project-space/
    │
    ├── [GNN Projects]
    │   ├── coastflow-gnn__project-space/
    │   ├── clothgnn__project-space/
    │   ├── hgnn-clothdyn__project-space/
    │   └── pegnn-deform__project-space/
    │
    ├── [NIF Projects]
    │   ├── geom-inr-motion__project-space/
    │   ├── clothgeom-nif__project-space/
    │   ├── nif-cloth4d__project-space/
    │   ├── nif-cloth4d-temporal__project-space/
    │   └── nif-cloth3d__project-space/
    │
    └── [Hybrid Projects]
        └── hgnn-nif-cloth__project-space/
```

### Project-Space Structure (Each Project)

```
<project-name>__project-space/
├── README.md                          # TRACKED - Project documentation
│
├── <project-name>/                    # TRACKED - Source code
│   ├── src/
│   ├── notebooks/
│   ├── configs/
│   ├── tests/
│   ├── requirements.txt
│   └── REGENERATE.md                  # How to recreate artifacts
│
├── docker/                            # TRACKED - Docker definitions
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── scripts/
│
├── docs/                              # TRACKED - Project-specific docs
│   └── *.md.txt                       # Research docs copied from research-docs/
│
└── docker-backups/                    # IGNORED - Local backups only
    ├── current_image.tar.gz           # Most recent Docker image
    ├── previous_image.tar.gz          # Auto-rotated safety backup
    ├── files/                         # Timestamped workspace backups
    │   ├── 2025-01-01_143022_session.tar.gz
    │   └── ...
    ├── archived/                      # Milestone images
    │   └── <milestone>_YYYY-MM-DD.tar.gz
    ├── .current_instance_id           # Active instance tracking
    ├── .last_saved                    # Last backup timestamp
    ├── .last_loaded                   # Last load timestamp
    └── .operations.log                # Operation history
```

### Remote Instance Layout (/workspace)

```
/workspace/                                    # Container workspace root
├── src/                                       # Source code (mirrors git)
├── notebooks/                                 # Jupyter notebooks
├── configs/                                   # Configuration files
├── data/                                      # Datasets (processed/generated, NOT raw downloads)
├── checkpoints/                               # Model weights
│   ├── best_model.pt
│   ├── final_model.pt
│   └── checkpoint_epoch_*.pt
├── results/                                   # Outputs
│   ├── metrics.json
│   ├── benchmark_summary.csv
│   └── plots/
├── logs/                                      # Training logs
└── REGENERATE.md                              # Documents how to recreate regenerable artifacts
```

### What Each Directory Contains

| Directory | Tracked | Purpose |
|-----------|---------|---------|
| `<project-name>/` | ✅ Yes | Source code, notebooks, configs |
| `docker/` | ✅ Yes | Dockerfile, docker-compose.yml, build scripts |
| `docs/` | ✅ Yes | Project-specific research docs |
| `README.md` | ✅ Yes | Project documentation |
| `docker-backups/` | ❌ No | Docker images, file backups, instance state |
| `research-docs/` (root) | ✅ Yes | Centralized research documentation |

---

## Project Categories

Projects are organized into four categories based on their primary approach:

### PINN Projects (Physics-Informed Neural Networks)
Neural networks with physics loss terms embedded in training.

| Project | Description |
|---------|-------------|
| `pinn-lite-foil` | Lightweight PINN for airfoil simulation |
| `maxwell-pinn-nif` | Maxwell equations with NIF integration |
| `GeoPINN-Manifold` | Geometric PINN on manifolds |
| `cell-path-pinns` | Cell path modeling with PINNs |
| `surfpinn` | Surface/free-surface PINNs |
| `wavepinn-nif` | Wave equation PINN with NIF |
| `WavePINN-NIF-ComplexMedia` | Wave propagation in complex media |

### GNN Projects (Graph Neural Networks)
Networks operating on graph-structured data.

| Project | Description |
|---------|-------------|
| `coastflow-gnn` | Coastal flow prediction |
| `clothgnn` | Cloth simulation with GNNs |
| `hgnn-clothdyn` | Hierarchical GNN for cloth dynamics |
| `pegnn-deform` | Position-encoded GNN for deformation |

### NIF Projects (Neural Implicit Functions)
Coordinate-based networks for implicit representations.

| Project | Description |
|---------|-------------|
| `geom-inr-motion` | Geometric INR for motion |
| `clothgeom-nif` | Cloth geometry with NIF |
| `nif-cloth4d` | 4D cloth representation |
| `nif-cloth4d-temporal` | Temporal 4D cloth |
| `nif-cloth3d` | 3D cloth representation |

### Hybrid Projects
Combinations of multiple approaches.

| Project | Description |
|---------|-------------|
| `hgnn-nif-cloth` | HGNN + NIF for cloth |

---

## Git Integration

### What Git Tracks

- All source code in `<project-name>/`
- Docker definitions in `docker/`
- Project-specific docs in `docs/`
- Project README.md files
- Root `research-docs/` folder
- This workflow documentation

### What Git Ignores

The root `.gitignore` must include:

```gitignore
# =============================================================================
# VAST.AI DOCKER BACKUPS - Never commit these
# =============================================================================
projects/*__project-space/docker-backups/

# =============================================================================
# RUNTIME ARTIFACTS - Ignored inside project-space folders
# =============================================================================
projects/*__project-space/**/volumes/
projects/*__project-space/**/runs/
projects/*__project-space/**/outputs/
projects/*__project-space/**/checkpoints/
projects/*__project-space/**/logs/
projects/*__project-space/**/tmp/
projects/*__project-space/**/cache/
projects/*__project-space/**/.cache/

# Compressed archives (backups)
projects/*__project-space/**/*.tar
projects/*__project-space/**/*.tar.gz

# =============================================================================
# LOCAL-ONLY DIRECTORIES
# =============================================================================
*.local/
_local/
**/*.local/
**/_local/

# =============================================================================
# PYTHON / JUPYTER
# =============================================================================
**/__pycache__/
**/*.pyc
**/.pytest_cache/
**/.mypy_cache/
**/.ruff_cache/
.ipynb_checkpoints/

# =============================================================================
# SECRETS
# =============================================================================
.env
.env.*
*.pem
*.key

# =============================================================================
# OS
# =============================================================================
.DS_Store
Thumbs.db
```

### Why Backups Are Not in Git

Docker images are 10-50GB each. File backups can be multiple gigabytes. Git is not designed for large binary files. Keeping backups outside git:

- Prevents repository bloat
- Avoids slow clone times
- Keeps git history clean
- Stores backups locally for free (no cloud storage fees)

---

## The Two Backup Types

Every session ends with **two backups**. Both are mandatory. No exceptions.

### 1. Docker Image Backup

**File**: `docker-backups/current_image.tar.gz`

**Contains**:
- Complete container filesystem
- All installed system packages
- All Python packages and environments
- All code (copy of what's in the container)
- All data, weights, checkpoints
- All configuration files
- Environment variables
- System-level changes

**Size**: 10-50GB compressed (varies by project)

**Rotation**:
- When saving new: `current_image.tar.gz` → `previous_image.tar.gz`
- New backup → `current_image.tar.gz`
- Always have two recent images

**Use case**: Primary restore method. Gives exact environment in minutes.

### 2. File Backup

**File**: `docker-backups/files/YYYY-MM-DD_HHMMSS_session.tar.gz`

**Contains**:
- Complete `/workspace` directory
- **Everything**: weights, data, cache, temp files, ALL files
- **No exclusions** - we want complete snapshots

**Size**: 500MB-10GB+ (depends on data and weights)

**Retention**: Keep ~15-20 most recent sessions

**Use cases**:
- Quick file extraction without loading full Docker image
- Emergency restore if Docker image corrupts
- Historical record of workspace state
- Redundancy layer

### Data Organization to Keep Backups Manageable

The "no exclusions" rule means file backups capture everything. To keep sizes reasonable:

- `/workspace/data/` should contain **processed/generated** data, not raw downloads
- Raw datasets that can be re-downloaded should be fetched on instance creation, not stored in backups
- Document download commands in `REGENERATE.md` so you can rebuild if needed

This preserves completeness while avoiding 50GB+ file backups full of re-downloadable data.

---

## Artifact Preservation Policy

The files most likely to be lost are produced artifacts (datasets, checkpoints, plots, metrics). This section defines exactly what must be saved, where it must live, and when it must be exported.

### Artifact Classes

#### A) Regenerable Artifacts (Allowed to regenerate)

Artifacts that can be recreated deterministically from committed code + committed config.

**Examples**:
- Synthetic datasets generated from a script + seed
- Benchmark plots (PNG/PDF) created from stored metrics
- Intermediate checkpoints that aren't "best" or "final"

**Policy**:
- Still included in file backups for convenience
- Loss is acceptable **only if** regeneration is documented in `REGENERATE.md`
- Must be deterministic (same code + config + seed = same output)

#### B) Costly / Irreplaceable Artifacts (Must be preserved)

Artifacts that are expensive (time/money) or impossible to recreate exactly.

**Examples**:
- `checkpoints/best_model.pt`
- `checkpoints/final_model.pt`
- Long-run experiment outputs (hours of GPU time)
- Final benchmark summaries (JSON/CSV)
- Any manually curated dataset

**Policy**:
- MUST be present in **at least one** backup before destroying an instance
- MUST exist locally after `save` finishes
- SHOULD be duplicated in both Docker image + file backup

### Artifact Directory Contract

All important outputs MUST be written under `/workspace` in one of these directories:

| Directory | Contents |
|-----------|----------|
| `/workspace/data/` | Datasets (processed/generated, not raw downloads) |
| `/workspace/checkpoints/` | Weights, best checkpoints, final models |
| `/workspace/results/` | Metrics (JSON/CSV), plots (PNG/PDF), evaluation summaries |
| `/workspace/logs/` | Training logs, stdout/stderr captures |

**Rule**: If a file matters, it must live in one of these directories.

This contract is about **organization discipline**. File backups capture all of `/workspace` regardless, but following this structure means:
- You always know where to find outputs
- Scripts can reliably locate artifacts
- Nothing gets scattered in random locations

### Time-Cost Threshold Rule

Any artifact that took **more than 10 minutes of GPU time** to produce must be present in a backup before the instance is destroyed.

| If this took >10 min | Then this must exist |
|----------------------|----------------------|
| Training | Checkpoints in `/workspace/checkpoints/` |
| Data generation | Dataset in `/workspace/data/` |
| Benchmarking | Metrics/plots in `/workspace/results/` |

### REGENERATE.md Convention

Each project should maintain a `/workspace/REGENERATE.md` file documenting how to recreate regenerable artifacts:

```markdown
# Regeneration Instructions

## Synthetic Dataset
python src/generate_data.py --seed 42 --n_samples 10000
Output: data/synthetic_train.npy

## Benchmark Plots
python src/plot_results.py --input results/metrics.json
Output: results/plots/*.png
```

This file should be committed to git (in the `<project-name>/` code folder) so it survives even if all backups are lost.

---

## Instance Lifecycle

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR MAC (Local)                         │
│                                                             │
│  ~/Dev/PINN-Experiments/projects/<project>__project-space/  │
│  └── docker-backups/                                        │
│      ├── current_image.tar.gz  ◄──────────┐                │
│      ├── previous_image.tar.gz            │                │
│      └── files/                           │  Download      │
│          └── session_backups.tar.gz  ◄────┤                │
│                                           │                │
└───────────────────────────────────────────┼────────────────┘
                      │                     │
                Upload│                     │
                      ▼                     │
┌─────────────────────────────────────────────────────────────┐
│                 VAST.AI INSTANCE (Remote)                   │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              DOCKER CONTAINER                         │ │
│  │                                                       │ │
│  │  /workspace/                                          │ │
│  │  ├── src/                                             │ │
│  │  ├── data/                                            │ │
│  │  ├── checkpoints/                                     │ │
│  │  ├── results/                                         │ │
│  │  └── logs/                                            │ │
│  │                                                       │ │
│  │  Your complete working environment                    │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  Instance runs base image, Docker runs INSIDE it            │
└─────────────────────────────────────────────────────────────┘
```

### Lifecycle States

```
[No Instance] ──create──► [Instance Running] ──destroy──► [No Instance]
                               │      ▲
                               │      │
                             load   save
                               │      │
                               ▼      │
                         [Working in Container]
```

---

## Daily Workflow

### Starting a Work Session

```bash
# 1. Create instance and load environment
./vastai-manage.sh create <project_name> --gpu RTX4090

# Script will:
# - Find suitable instances
# - Create and wait for ready
# - Prompt to load existing Docker image
# - Provide SSH connection info

# 2. Connect and work
ssh <connection_info>
# ... do your work ...
```

### Ending a Work Session

```bash
# 1. Complete the pre-save checklist (see section below)

# 2. Save both backups (MANDATORY)
./vastai-manage.sh save <project_name>

# Script will:
# - Commit container to image
# - Compress and download Docker image
# - Rotate current → previous
# - Create timestamped file backup
# - Download file backup

# 3. Destroy instance (stops billing)
./vastai-manage.sh destroy <project_name>

# Script will:
# - Verify backup was saved today
# - Require confirmation
# - Destroy instance
# - Confirm billing stopped
```

### Quick Reference

| Action | Command |
|--------|---------|
| Start new session | `./vastai-manage.sh create <project>` |
| Load saved environment | `./vastai-manage.sh load <project>` |
| Save before stopping | `./vastai-manage.sh save <project>` |
| Stop and stop billing | `./vastai-manage.sh destroy <project>` |
| Check status | `./vastai-manage.sh status <project>` |
| Archive milestone | `./vastai-manage.sh archive <project> <name>` |
| Clean old backups | `./vastai-manage.sh cleanup <project>` |
| List all projects | `./vastai-manage.sh list-projects` |

---

## Multi-Project Handling

Each project tracks its own instance independently via `.current_instance_id`.

### Checking All Instances

```bash
# See all running instances across all projects
./vastai-manage.sh list-instances

# See status of all projects
./vastai-manage.sh dashboard
```

### Running Multiple Projects Simultaneously

You CAN run multiple projects on different instances, but:

⚠️ **Each running instance costs money** — Destroy when not actively using

```bash
# Check what's running
./vastai-manage.sh list-instances

# Destroy specific project's instance
./vastai-manage.sh destroy hgnn-clothdyn
```

### Switching Between Projects

```bash
# Save current project
./vastai-manage.sh save project-a

# Destroy to stop billing
./vastai-manage.sh destroy project-a

# Start different project
./vastai-manage.sh create project-b
```

---

## Script Reference

### vastai-manage.sh Commands

#### `create <project_name> [options]`

Creates a new Vast.ai instance.

Options:
- `--gpu <type>` - GPU model (default: RTX4090)
- `--min-ram <GB>` - Minimum RAM (default: 32)
- `--max-price <$/hr>` - Maximum hourly rate (default: 2.0)
- `--min-reliability <score>` - Minimum reliability (default: 0.95)

#### `load <project_name>`

Uploads and loads `current_image.tar.gz` to the running instance.

#### `save <project_name>`

Creates both backups:
1. Commits container → compresses → downloads Docker image
2. Creates timestamped file backup of /workspace

Automatically rotates `current_image.tar.gz` → `previous_image.tar.gz`.

#### `destroy <project_name>`

Destroys the instance. Has safety checks:
- Warns if no backup saved today
- Requires explicit confirmation

#### `archive <project_name> <milestone_name>`

Copies `current_image.tar.gz` to `archived/<milestone_name>_YYYY-MM-DD.tar.gz`.

Use before:
- Major refactoring
- Changing approaches
- Risky experiments

#### `cleanup <project_name> [--keep N]`

Removes old file backups, keeping the N most recent (default: 15).

#### `status <project_name>`

Shows:
- Current instance status
- Last backup timestamps
- Backup sizes
- Number of file backups

#### `list-projects`

Lists all available projects in the repository.

#### `emergency-restore <project_name>`

For when Docker image is corrupted:
1. Creates fresh instance
2. Uploads most recent file backup
3. Extracts to /workspace
4. Lists packages that may need reinstalling

---

## Long-Running Job Safety

For jobs expected to run longer than ~2 hours:

### Checkpoint Strategy

- Save model checkpoints at fixed intervals (every N epochs or every N minutes)
- Always save when validation improves ("best" checkpoint)
- Keep at least: `best_model.pt`, `latest_model.pt`, `final_model.pt`

### Incremental Results

- Write metrics incrementally (append to JSONL or CSV)
- Don't accumulate everything in memory and write once at the end
- If job crashes, you still have partial results

### Periodic File Backups

For very long runs (4+ hours), consider mid-session file backups:

```bash
# Quick file-only backup (no Docker image)
./vastai-manage.sh save-files <project_name>
```

This is faster than a full save and provides a checkpoint.

### Recommended Timing

| Run Length | Checkpoint Interval | File Backup |
|------------|---------------------|-------------|
| < 2 hours | Every 30 min | End of session only |
| 2-4 hours | Every 15-30 min | Once mid-session |
| 4-8 hours | Every 15 min | Every 2 hours |
| 8+ hours | Every 10 min | Every 2 hours |

**Never** wait until the end of an 8-12 hour run to save anything.

---

## Pre-Save Checklist

Before running `./vastai-manage.sh save <project_name>`:

- [ ] **No active writes**: Jobs finished or safely checkpointed (no half-written files)
- [ ] **Checkpoints exist**: `ls /workspace/checkpoints/` shows best/final models
- [ ] **Results exist**: `ls /workspace/results/` shows metrics/plots
- [ ] **Notes updated**: Important observations added to README or logs
- [ ] **Config committed**: Changes either committed to git OR intentionally left for later
- [ ] **Quick verify**: `du -sh /workspace/*` to sanity-check sizes

### Quick Verification Command

Run this before saving to verify artifacts exist:

```bash
echo "=== Checkpoints ===" && ls -la /workspace/checkpoints/ && \
echo "=== Results ===" && ls -la /workspace/results/ && \
echo "=== Sizes ===" && du -sh /workspace/*
```

---

## Docker Image Hygiene

### What Docker Images Are For

- ✅ Environment reproducibility
- ✅ Fast restore of complete workspace
- ✅ Emergency recovery

### What NOT to Store in Docker Images

- ❌ **API keys / secrets**: Use `.env` locally, never bake into images
- ❌ **Huge raw datasets**: Re-download on instance creation instead
- ❌ **Personal credentials**: SSH keys, tokens, passwords
- ❌ **Temporary files**: Build artifacts, pip cache (unless needed)

### Size Management

If Docker images grow too large (>50GB):

1. Check what's consuming space: `du -sh /workspace/* | sort -h`
2. Move raw downloadable data out of `/workspace/data/`
3. Clean pip/conda caches: `pip cache purge && conda clean -a`
4. Remove intermediate checkpoints, keep only best/final

---

## Emergency Procedures

### Docker Image Won't Load

1. Try `previous_image.tar.gz`:
   ```bash
   cp docker-backups/previous_image.tar.gz docker-backups/current_image.tar.gz
   ./vastai-manage.sh load <project>
   ```

2. If both images fail, use emergency restore:
   ```bash
   ./vastai-manage.sh emergency-restore <project>
   ```

3. After emergency restore, reinstall system packages manually

### Instance Crashed Before Saving

Work may be lost. This is why we:
- Save frequently during long sessions
- Never skip the save step
- Keep multiple file backups

Prevention: Save every few hours during long sessions, not just at the end.

### Corrupted File Backup

File backups are redundant. Use the Docker image instead—it contains everything.

### Ran Out of Disk Space on Mac

1. Run cleanup to remove old file backups:
   ```bash
   ./vastai-manage.sh cleanup <project> --keep 10
   ```

2. Delete old archived images you no longer need

3. Consider keeping only current (not previous) Docker image temporarily

---

## Restore Verification

**Untested backups are Schrödinger's backups.** You don't know if they work until you try.

### Monthly Verification Procedure

At least once per month per active project:

1. **Create fresh instance**
   ```bash
   ./vastai-manage.sh create <project> --gpu RTX4090
   ```

2. **Load the backup**
   ```bash
   ./vastai-manage.sh load <project>
   ```

3. **Run minimal verification**
   ```bash
   # Check Python environment
   python -c "import torch; print(torch.__version__)"
   
   # Load a checkpoint
   python -c "import torch; m = torch.load('/workspace/checkpoints/best_model.pt'); print('OK')"
   
   # Run one inference step (project-specific)
   python src/inference.py --checkpoint /workspace/checkpoints/best_model.pt --quick-test
   ```

4. **Verify files exist**
   ```bash
   ls -la /workspace/checkpoints/
   ls -la /workspace/results/
   ```

5. **Destroy test instance**
   ```bash
   ./vastai-manage.sh destroy <project>
   ```

### Verification Log

Keep a simple log in your project:

```markdown
## Backup Verification Log

| Date | Image | Result | Notes |
|------|-------|--------|-------|
| 2025-01-15 | current | ✅ Pass | Loaded in 3 min, inference OK |
| 2024-12-20 | current | ✅ Pass | |
```

---

## Recovery of Lost Artifacts

If an instance is destroyed and artifacts are missing:

### Step 1: Classify the Missing Files

| Type | Action |
|------|--------|
| Regenerable | Regenerate using `REGENERATE.md` |
| Costly/Irreplaceable | Accept loss, plan re-run |

### Step 2: Regeneration Procedure

1. Restore environment (Docker image preferred)
2. Checkout the git commit used for the original run
3. Re-run generation scripts:
   ```bash
   # Data
   python src/generate_data.py --seed 42
   
   # Training (if checkpoints lost)
   python src/train.py --config configs/experiment.yaml
   
   # Benchmarks
   python src/benchmark.py --checkpoint checkpoints/best_model.pt
   ```
4. Save outputs into `/workspace/{data,checkpoints,results}`
5. Run full backup before destroying

### Step 3: Prevention

To avoid future losses:
- Follow the Artifact Directory Contract
- Follow the Time-Cost Threshold Rule
- Save periodically during long runs
- Maintain `REGENERATE.md` for all regenerable artifacts

---

## Storage Management

### Expected Disk Usage

| Item | Size | Quantity | Total |
|------|------|----------|-------|
| Docker images | 10-50GB | 2 (current + previous) | 20-100GB |
| File backups | 0.5-10GB | 15-20 sessions | 7-200GB |
| Archived images | 10-50GB | 2-5 milestones | 20-250GB |

**Recommended free space**: 150-300GB per active project

**Note**: With 18+ projects in this repo, you likely won't have active backups for all. Focus on 2-3 active projects at a time.

### Cleanup Strategy

**File backups**: Keep 15-20 recent sessions. Delete older ones monthly.

**Docker images**: Keep only current + previous. Archive before major changes.

**Archived images**: Keep meaningful milestones only:
- Before major refactors
- Working versions before experiments
- Release points

### Checking Usage

```bash
# Check docker-backups size for all projects
du -sh ~/Dev/PINN-Experiments/projects/*__project-space/docker-backups/

# Detailed breakdown for one project
du -sh ~/Dev/PINN-Experiments/projects/<project>__project-space/docker-backups/*

# List file backups with sizes
ls -lh ~/Dev/PINN-Experiments/projects/<project>__project-space/docker-backups/files/

# Find largest backup directories
find ~/Dev/PINN-Experiments/projects -name "docker-backups" -exec du -sh {} \; | sort -h
```

---

## Cost Tracking

### Monitoring Spending

- **Dashboard**: https://cloud.vast.ai/console/billing/
- **CLI**: `vastai show invoices`
- **Per-session target**: <$5 (~4 hours @ $1.25/hr for RTX 4090)

### Cost Optimization Tips

1. **Destroy immediately after saving** — Every hour counts
2. **Use `save-files` for quick mid-session backups** — Much faster than full saves
3. **Archive only meaningful milestones** — Not every experiment
4. **Choose GPUs wisely**:
   | GPU | Typical $/hr | Best For |
   |-----|--------------|----------|
   | RTX 4090 | $0.80-1.50 | Most training |
   | L40S | $1.00-1.80 | Larger models |
   | A100 | $1.50-3.00 | Very large models |

### Tracking by Project

Use the dashboard to see which projects have active instances:

```bash
./vastai-manage.sh dashboard
```

---

## Best Practices

### Always

- ✅ Save both backups at end of every session
- ✅ Complete the pre-save checklist before saving
- ✅ Verify backups completed before destroying instance
- ✅ Archive before risky experiments
- ✅ Keep `previous_image.tar.gz` as safety net
- ✅ Follow the Artifact Directory Contract
- ✅ Maintain `REGENERATE.md` for regenerable artifacts

### Never

- ❌ Destroy instance without saving
- ❌ Skip file backup because "Docker image has everything"
- ❌ Commit `docker-backups/` to git
- ❌ Let file backups accumulate indefinitely
- ❌ Store secrets/credentials in Docker images
- ❌ Wait until end of 8+ hour run to save anything

### Recommended Habits

1. **Save frequently during long sessions** — Every 2 hours for runs over 4 hours
2. **Archive before experiments** — Name descriptively: `before_new_loss_function`
3. **Document manual setup steps** — System packages, environment variables
4. **Test restores monthly** — Verify backups actually work
5. **Clean up monthly** — Remove old file backups and unnecessary archives
6. **Keep REGENERATE.md current** — Update when adding new generated artifacts

---

## Troubleshooting

### "Instance not found" errors

- Instance may have been terminated by Vast.ai (rare)
- Check `.current_instance_id` matches actual instance
- Run `./vastai-manage.sh list-instances` to see all instances

### Slow uploads/downloads

- Large Docker images take time (10-50GB)
- Expect 15-45 minutes for full save cycle
- File backups are faster (usually 5-15 minutes)

### "Permission denied" on remote

- Ensure SSH key is configured for Vast.ai
- Check instance is fully booted (not still initializing)

### Container won't start from loaded image

- Image may be corrupted during transfer
- Re-upload and try again
- If persists, use emergency-restore with file backup

### Backup file sizes growing unexpectedly

- Check `/workspace/data/` for raw downloads that should be excluded
- Check for accumulated logs or intermediate checkpoints
- Run `du -sh /workspace/* | sort -h` to find the culprit

---

## Version History

| Date | Change |
|------|--------|
| 2025-01-03 | Updated with actual project structure, added project categories |
| 2025-01-03 | Added artifact preservation policy, pre-save checklist, restore verification |
| 2025-01-02 | Initial workflow documentation |

---

## Related Files

- `vastai-manage.sh` — Instance management script
- `.gitignore` — Git ignore rules including docker-backups/
- `research-docs/` — Centralized research documentation
- `projects/<project>__project-space/docker/Dockerfile` — Per-project Docker definitions
- `projects/<project>__project-space/<project>/REGENERATE.md` — Artifact regeneration instructions
