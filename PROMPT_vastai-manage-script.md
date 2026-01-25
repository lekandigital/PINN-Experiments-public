> **Note**: This prompt has been incorporated into `CLAUDE_CODE_MONOREPO_SETUP.md`. 
> Keep this file for reference but use the consolidated prompt for execution.

---

# Prompt: Generate vastai-manage.sh Script

## Context

I use Vast.ai CLI to rent GPU instances for machine learning work. I run Docker containers INSIDE the rented Vast.ai instances (not using Vast.ai's Docker rental mode). My workflow is designed to minimize costs: create instance → load environment → work → save backups → destroy instance.

## My Environment

- **Local machine**: Mac with Docker CLI installed and authenticated
- **Vast.ai CLI**: installed and authenticated locally
- **Repository root**: `~/Dev/PINN-Experiments/`

## Project Structure

```
~/Dev/PINN-Experiments/                        # Git repo root
├── .git/
├── .gitignore
├── README.md
├── WORKFLOW.md
├── vastai-manage.sh                           # THIS SCRIPT
│
├── research-docs/                             # Centralized research docs
│   └── ...
│
└── projects/                                  # All projects
    ├── pinn-lite-foil__project-space/
    ├── maxwell-pinn-nif__project-space/
    ├── GeoPINN-Manifold__project-space/
    ├── cell-path-pinns__project-space/
    ├── surfpinn__project-space/
    ├── wavepinn-nif__project-space/
    ├── WavePINN-NIF-ComplexMedia__project-space/
    ├── coastflow-gnn__project-space/
    ├── clothgnn__project-space/
    ├── hgnn-clothdyn__project-space/
    ├── pegnn-deform__project-space/
    ├── geom-inr-motion__project-space/
    ├── clothgeom-nif__project-space/
    ├── nif-cloth4d__project-space/
    ├── nif-cloth4d-temporal__project-space/
    ├── nif-cloth3d__project-space/
    └── hgnn-nif-cloth__project-space/
```

Each project-space contains:
```
<project-name>__project-space/
├── README.md                          # Project documentation
├── <project-name>/                    # Source code
├── docker/                            # Dockerfile, docker-compose.yml
├── docs/                              # Project-specific docs
└── docker-backups/                    # IGNORED by git - backups here
    ├── current_image.tar.gz           # Most recent Docker image
    ├── previous_image.tar.gz          # Auto-rotated backup
    ├── files/                         # Timestamped workspace backups
    │   └── YYYY-MM-DD_HHMMSS_session.tar.gz
    ├── archived/                      # Milestone images
    │   └── <milestone_name>_YYYY-MM-DD.tar.gz
    ├── .current_instance_id           # Active instance tracking
    ├── .last_saved                    # Last backup timestamp
    ├── .last_loaded                   # Last load timestamp
    └── .operations.log                # Operation history
```

## Remote Instance Layout

When a Vast.ai instance is running:
- Instance runs a base image (e.g., PyTorch/CUDA)
- I start a Docker container INSIDE the instance
- Workspace inside container: `/workspace`
- The Docker container contains my complete environment: code, data, weights, packages, configs

## Backup Strategy (CRITICAL)

**Two backups, EVERY session, NO exceptions:**

1. **Docker Image Backup** (`current_image.tar.gz`)
   - Complete snapshot of the Docker container
   - Includes: all installed packages, code, data, weights, configs, environment variables, system changes
   - Typically 10-50GB compressed
   - Auto-rotation: current → previous when saving new

2. **File Backup** (`files/YYYY-MM-DD_HHMMSS_session.tar.gz`)
   - Complete tarball of `/workspace` directory
   - **NO EXCLUSIONS**: include everything - weights, checkpoints, datasets, cache, temp files, ALL files
   - Timestamped, accumulates over sessions
   - Keep ~15-20 recent backups

## Script Requirements

Create `vastai-manage.sh` with these subcommands:

### `./vastai-manage.sh create <project_name> [options]`

1. Validate project exists at `~/Dev/PINN-Experiments/projects/<project_name>__project-space/`
2. Create `docker-backups/` subdirectories if they don't exist (`files/`, `archived/`)
3. Search for available Vast.ai instances with filters:
   - `--gpu <type>` (default: RTX4090)
   - `--min-ram <GB>` (default: 32)
   - `--max-price <$/hr>` (default: 2.0)
   - `--min-reliability <score>` (default: 0.95)
4. Display top 5 options, let user select
5. Create instance with selected machine
6. Wait for instance to be ready (poll status)
7. Display instance ID, SSH command, and connection info
8. Save instance ID to `docker-backups/.current_instance_id`
9. Check if `current_image.tar.gz` exists:
   - If yes: prompt "Load existing Docker image? (y/n)"
   - If yes: automatically run `load` subcommand
   - If no: display instructions for manual setup

### `./vastai-manage.sh load <project_name>`

1. Read instance ID from `docker-backups/.current_instance_id`
2. Verify instance is running
3. Upload `current_image.tar.gz` to instance via SCP
4. SSH into instance and execute:
   - `docker load -i <uploaded_image.tar.gz>`
   - Stop any running containers
   - Start new container from loaded image with `/workspace` mounted appropriately
5. Display success message and SSH command to enter the container
6. Record load timestamp to `docker-backups/.last_loaded`

### `./vastai-manage.sh save <project_name>`

1. Read instance ID from `docker-backups/.current_instance_id`
2. Verify instance is running
3. SSH into instance and execute:
   
   **Docker image backup:**
   - Get running container ID
   - `docker commit <container_id> <project_name>:latest`
   - `docker save <project_name>:latest | gzip > /tmp/docker_backup.tar.gz`
   
   **File backup:**
   - `tar czf /tmp/workspace_backup.tar.gz -C /workspace .` (NO exclusions)

4. Download both files to local Mac:
   - Docker image → temporary location first
   - File backup → `docker-backups/files/YYYY-MM-DD_HHMMSS_session.tar.gz`

5. Rotate Docker images:
   - If `current_image.tar.gz` exists → move to `previous_image.tar.gz`
   - Move downloaded image → `current_image.tar.gz`

6. Record save timestamp to `docker-backups/.last_saved`
7. Display:
   - Backup sizes
   - Total time taken
   - Confirmation message
   - Number of file backups now stored

### `./vastai-manage.sh save-files <project_name>`

Quick file-only backup for mid-session saves during long runs:
1. Read instance ID from `docker-backups/.current_instance_id`
2. Create file backup only (skip Docker image)
3. Download to `docker-backups/files/YYYY-MM-DD_HHMMSS_session.tar.gz`
4. Much faster than full save

### `./vastai-manage.sh destroy <project_name>`

1. Read instance ID from `docker-backups/.current_instance_id`
2. **SAFETY CHECK**: Compare `.last_saved` timestamp with current time
   - If no save today: WARN "No backup saved this session!"
   - Require user to type "destroy without backup" to proceed without backup
   - Otherwise require typing "yes" to confirm
3. Destroy instance via Vast.ai CLI
4. Remove `.current_instance_id` file
5. Display confirmation that billing has stopped

### `./vastai-manage.sh archive <project_name> <milestone_name>`

1. Verify `current_image.tar.gz` exists
2. Copy (not move) to `docker-backups/archived/<milestone_name>_YYYY-MM-DD.tar.gz`
3. Display archive location and size
4. List all archived images

### `./vastai-manage.sh cleanup <project_name> [--keep N]`

1. List all file backups in `docker-backups/files/` with dates and sizes
2. Default: keep newest 15 backups
3. Show which backups would be deleted
4. Prompt for confirmation
5. Delete old backups
6. Display disk space recovered

### `./vastai-manage.sh emergency-restore <project_name>`

1. For use when Docker image is corrupted
2. Run `create` to get a fresh instance
3. Find most recent file backup in `docker-backups/files/`
4. Upload and extract to `/workspace`
5. Display warning: "Environment restored from file backup. System packages may need manual reinstall."
6. List common packages that typically need reinstalling

### `./vastai-manage.sh status <project_name>`

1. Show current instance status (if any)
2. Show last backup timestamps
3. Show backup sizes
4. Show count of file backups and total size
5. Show count of archived images

### `./vastai-manage.sh list-projects`

1. Scan `~/Dev/PINN-Experiments/projects/` for `*__project-space/` directories
2. List all projects with:
   - Project name
   - Whether docker-backups exist
   - Last backup date (if any)
   - Current instance status (if any)

### `./vastai-manage.sh list-instances`

1. Show all running Vast.ai instances
2. Match to projects if possible (via `.current_instance_id` files)

## Error Handling Requirements

- All subcommands should validate arguments and show usage on error
- Network failures should retry 3 times with exponential backoff
- Partial uploads/downloads should be detected and cleaned up
- SSH connection failures should show helpful debugging info
- All operations should be idempotent where possible

## Output Requirements

- Use clear status messages with timestamps
- Show progress for long operations (upload/download)
- Use colors for success (green), warning (yellow), error (red)
- Estimate time remaining for transfers based on file size

## Additional Features

- `./vastai-manage.sh help` - show all commands and usage
- Log all operations to `docker-backups/.operations.log` with timestamps

## Shell Compatibility

- Use bash (#!/bin/bash)
- Compatible with macOS default shell
- Use standard Unix tools (scp, ssh, tar, gzip, etc.)

## The project I'm working with is: `[PROJECT_NAME]`

Please generate the complete `vastai-manage.sh` script following these specifications.
