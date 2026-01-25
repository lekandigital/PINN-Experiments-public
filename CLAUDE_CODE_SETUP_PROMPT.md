# Claude Code Task: PINN-Experiments Setup & vastai-manage.sh Generation

## Context

I have a PINN-Experiments monorepo at `~/Dev/PINN-Experiments/` that needs some cleanup and a proper `vastai-manage.sh` script generated.

## Tasks to Complete

### Task 1: Set up Vast.ai API Key

Save this API key to both locations:
- `~/.vast_api_key`
- `~/Dev/PINN-Experiments/.secrets/vast_api_key`

API Key: `REDACTED_API_KEY`

Set permissions to 600 on both files.

Verify it works by running: `vastai show user`

### Task 2: Fix .gitignore Duplicates

The file `~/Dev/PINN-Experiments/.gitignore` has duplicate entries at the end. Remove the duplicate "SECRETS" section (the last 5-6 lines that duplicate `.secrets/`, `*.pem`, `*.key`).

### Task 3: Compress Extracted Backup Folders

In `~/Dev/PINN-Experiments/projects/*/docker-backups/files/` there are extracted folders that should be compressed tar.gz files. 

For each directory (not file) in these locations:
1. Create a .tar.gz of the directory
2. Remove the original directory after successful compression

### Task 4: Create .env.example

Create `~/Dev/PINN-Experiments/.secrets.example/.env.example` with template content for VAST_API_KEY, HF_TOKEN, and WANDB_API_KEY.

### Task 5: Generate Complete vastai-manage.sh

Replace `~/Dev/PINN-Experiments/vastai-manage.sh` with a fully functional script that implements:

**Commands needed:**
- `create <project> [--gpu TYPE]` - Search instances, create one, wait for ready, save instance ID, prompt to load existing image
- `load <project>` - Upload and load current_image.tar.gz to running instance
- `save <project>` - Commit container to image, create file backup, download both, rotate current→previous
- `save-files <project>` - Quick file-only backup (no Docker image)
- `destroy <project>` - Safety check for backup, destroy instance, clear instance ID
- `archive <project> <milestone>` - Copy current image to archived/
- `cleanup <project> [--keep N]` - Remove old file backups, keep N most recent (default 15)
- `status <project>` - Show instance status, backup timestamps, sizes
- `list-projects` - List all *__project-space directories
- `list-instances` - Show all running Vast.ai instances
- `emergency-restore <project>` - Restore from file backup when Docker image corrupted
- `help` - Show usage

**Project structure:**
```
~/Dev/PINN-Experiments/
├── projects/
│   └── <name>__project-space/
│       └── docker-backups/
│           ├── current_image.tar.gz
│           ├── previous_image.tar.gz
│           ├── files/
│           │   └── YYYY-MM-DD_HHMMSS_session.tar.gz
│           ├── archived/
│           ├── .current_instance_id
│           ├── .last_saved
│           └── .last_loaded
```

**Remote instance workspace:** `/workspace/` with subdirs: data, checkpoints, results, logs

**Requirements:**
- Use vastai CLI commands (vastai search offers, vastai create instance, vastai destroy instance, etc.)
- Use scp for file transfers
- Use ssh for remote commands
- Color output (green=success, yellow=warning, red=error)
- Safety checks before destroy (warn if no backup today)
- Auto-rotation of Docker images (current → previous)
- Timestamps on all operations
- Log to docker-backups/.operations.log

### Task 6: Verify Setup

After completing all tasks:
1. Run `vastai show user` to confirm API key works
2. Run `./vastai-manage.sh help` to confirm script works
3. Run `./vastai-manage.sh list-projects` to show available projects
4. Show git status to confirm no untracked secrets

## Important Notes

- The script runs on macOS
- Docker containers run INSIDE the Vast.ai instances (not Vast.ai's Docker rental mode)
- File backups should have NO exclusions - capture everything in /workspace
- Never commit .secrets/ to git
