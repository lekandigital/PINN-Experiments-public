#!/bin/bash
# =============================================================================
# Vast.ai Instance Management Script for PINN-Experiments
# =============================================================================
# Usage: ./vastai-manage.sh <command> [project-name] [options]
# 
# Complete management of Vast.ai GPU instances for ML research projects.
# Handles instance creation, Docker image backup/restore, file backups,
# and project lifecycle management.
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT=~/Dev/PINN-Experiments
SECRETS_DIR="$PROJECT_ROOT/.secrets"
DEFAULT_GPU="RTX_4090"
DEFAULT_DISK_GB=100
MAX_RETRIES=3
RETRY_DELAY=5

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_operation() {
    local project="$1"
    local operation="$2"
    local status="$3"
    local backup_dir="${PROJECT_ROOT}/projects/${project}__project-space/docker-backups"
    local log_file="${backup_dir}/.operations.log"
    
    mkdir -p "$backup_dir"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $operation: $status" >> "$log_file"
}

check_api_key() {
    # Check for API key in multiple locations
    if [[ -f ~/.vast_api_key ]]; then
        return 0
    elif [[ -f "$SECRETS_DIR/vast_api_key" ]]; then
        # Copy to expected location
        cp "$SECRETS_DIR/vast_api_key" ~/.vast_api_key
        chmod 600 ~/.vast_api_key
        return 0
    else
        log_error "Vast.ai API key not found!"
        echo ""
        echo "Run the configuration script:"
        echo "  ./configure-keys.sh"
        echo ""
        echo "Or manually create ~/.vast_api_key with your API key"
        exit 1
    fi
}

get_ssh_key() {
    if [[ -f ~/.ssh/id_ed25519_vastai ]]; then
        echo ~/.ssh/id_ed25519_vastai
    elif [[ -f ~/.ssh/id_ed25519 ]]; then
        echo ~/.ssh/id_ed25519
    elif [[ -f ~/.ssh/id_rsa ]]; then
        echo ~/.ssh/id_rsa
    else
        echo ""
    fi
}

validate_project() {
    local project="$1"
    local project_dir="${PROJECT_ROOT}/projects/${project}__project-space"
    
    if [[ -z "$project" ]]; then
        log_error "Project name required"
        exit 1
    fi
    
    if [[ ! -d "$project_dir" ]]; then
        log_error "Project not found: $project"
        echo "Available projects:"
        list_projects
        exit 1
    fi
}

get_instance_id() {
    local project="$1"
    local id_file="${PROJECT_ROOT}/projects/${project}__project-space/docker-backups/.current_instance_id"
    
    if [[ -f "$id_file" ]]; then
        cat "$id_file"
    else
        echo ""
    fi
}

save_instance_id() {
    local project="$1"
    local instance_id="$2"
    local backup_dir="${PROJECT_ROOT}/projects/${project}__project-space/docker-backups"
    
    mkdir -p "$backup_dir"
    echo "$instance_id" > "${backup_dir}/.current_instance_id"
}

clear_instance_id() {
    local project="$1"
    local id_file="${PROJECT_ROOT}/projects/${project}__project-space/docker-backups/.current_instance_id"
    
    if [[ -f "$id_file" ]]; then
        rm "$id_file"
    fi
}

get_instance_ssh_info() {
    local instance_id="$1"
    
    # Get instance details
    local info=$(vastai show instance "$instance_id" --raw 2>/dev/null)
    if [[ -z "$info" ]]; then
        return 1
    fi
    
    local ssh_host=$(echo "$info" | jq -r '.ssh_host // empty')
    local ssh_port=$(echo "$info" | jq -r '.ssh_port // empty')
    
    if [[ -n "$ssh_host" && -n "$ssh_port" ]]; then
        echo "${ssh_host}:${ssh_port}"
    else
        return 1
    fi
}

wait_for_instance() {
    local instance_id="$1"
    local max_wait=300  # 5 minutes
    local elapsed=0
    
    log_info "Waiting for instance $instance_id to be ready..."
    
    while [[ $elapsed -lt $max_wait ]]; do
        local status=$(vastai show instance "$instance_id" --raw 2>/dev/null | jq -r '.actual_status // empty')
        
        if [[ "$status" == "running" ]]; then
            log_success "Instance is running!"
            return 0
        elif [[ "$status" == "exited" || "$status" == "error" ]]; then
            log_error "Instance failed to start (status: $status)"
            return 1
        fi
        
        echo -n "."
        sleep 10
        elapsed=$((elapsed + 10))
    done
    
    echo ""
    log_error "Timeout waiting for instance"
    return 1
}

retry_command() {
    local cmd="$1"
    local retries=0
    
    while [[ $retries -lt $MAX_RETRIES ]]; do
        if eval "$cmd"; then
            return 0
        fi
        retries=$((retries + 1))
        if [[ $retries -lt $MAX_RETRIES ]]; then
            log_warn "Command failed, retrying in ${RETRY_DELAY}s... (attempt $((retries + 1))/$MAX_RETRIES)"
            sleep $((RETRY_DELAY * retries))  # Exponential backoff
        fi
    done
    
    log_error "Command failed after $MAX_RETRIES attempts"
    return 1
}

format_size() {
    local bytes="$1"
    if [[ $bytes -ge 1073741824 ]]; then
        echo "$(echo "scale=1; $bytes / 1073741824" | bc) GB"
    elif [[ $bytes -ge 1048576 ]]; then
        echo "$(echo "scale=1; $bytes / 1048576" | bc) MB"
    elif [[ $bytes -ge 1024 ]]; then
        echo "$(echo "scale=1; $bytes / 1024" | bc) KB"
    else
        echo "$bytes B"
    fi
}

# =============================================================================
# COMMANDS
# =============================================================================

show_help() {
    echo -e "${BOLD}Vast.ai Instance Manager for PINN-Experiments${NC}"
    echo ""
    echo "Usage: ./vastai-manage.sh <command> [project-name] [options]"
    echo ""
    echo -e "${CYAN}Instance Management:${NC}"
    echo "  create <project> [--gpu TYPE]   Create new instance for project"
    echo "  load <project>                  Upload and load Docker image to instance"
    echo "  destroy <project>               Destroy instance (with safety checks)"
    echo ""
    echo -e "${CYAN}Backup & Restore:${NC}"
    echo "  save <project>                  Save Docker image + file backup"
    echo "  save-files <project>            Quick file-only backup"
    echo "  archive <project> <milestone>   Archive current image to milestone"
    echo "  emergency-restore <project>     Restore from file backup"
    echo "  cleanup <project> [--keep N]    Remove old backups (default: keep 15)"
    echo ""
    echo -e "${CYAN}Status & Monitoring:${NC}"
    echo "  status <project>                Show project status and backups"
    echo "  list-projects                   List all projects with backup status"
    echo "  list-instances                  Show all running Vast.ai instances"
    echo "  dashboard                       Overview of all projects"
    echo ""
    echo -e "${CYAN}Configuration:${NC}"
    echo "  configure-keys                  Run interactive key setup"
    echo "  help                            Show this help"
    echo ""
    echo -e "${CYAN}Examples:${NC}"
    echo "  ./vastai-manage.sh create hgnn-clothdyn --gpu RTX_4090"
    echo "  ./vastai-manage.sh save pinn-lite-foil"
    echo "  ./vastai-manage.sh dashboard"
}

# -----------------------------------------------------------------------------
# create <project> [--gpu TYPE]
# -----------------------------------------------------------------------------
cmd_create() {
    local project="$1"
    local gpu_type="$DEFAULT_GPU"
    
    # Parse options
    shift
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --gpu)
                gpu_type="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    validate_project "$project"
    check_api_key
    
    local backup_dir="${PROJECT_ROOT}/projects/${project}__project-space/docker-backups"
    local existing_id=$(get_instance_id "$project")
    
    if [[ -n "$existing_id" ]]; then
        log_warn "Project already has instance: $existing_id"
        read -p "Destroy existing instance first? [y/N]: " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            cmd_destroy "$project"
        else
            exit 1
        fi
    fi
    
    echo -e "${BLUE}=============================================="
    echo "  Creating Instance for: $project"
    echo -e "==============================================${NC}"
    echo ""
    
    # Search for instances
    log_info "Searching for $gpu_type instances..."
    echo ""
    
    # Format GPU name for search (replace _ with space for display)
    local gpu_search=$(echo "$gpu_type" | tr '_' ' ')
    
    vastai search offers "gpu_name=$gpu_search rentable=true verified=true disk_space>=$DEFAULT_DISK_GB" \
        --order dph_total --limit 10
    
    echo ""
    read -p "Enter offer ID to rent (or 'q' to quit): " offer_id
    
    if [[ "$offer_id" == "q" || -z "$offer_id" ]]; then
        log_info "Cancelled"
        exit 0
    fi
    
    # Create instance
    log_info "Creating instance from offer $offer_id..."
    
    local create_output=$(vastai create instance "$offer_id" \
        --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel \
        --disk $DEFAULT_DISK_GB \
        --ssh \
        --direct \
        2>&1)
    
    local instance_id=$(echo "$create_output" | grep -oE '[0-9]+' | head -1)
    
    if [[ -z "$instance_id" ]]; then
        log_error "Failed to create instance"
        echo "$create_output"
        exit 1
    fi
    
    log_success "Instance created: $instance_id"
    save_instance_id "$project" "$instance_id"
    log_operation "$project" "CREATE" "instance_id=$instance_id"
    
    # Wait for instance
    if wait_for_instance "$instance_id"; then
        echo ""
        log_success "Instance ready!"
        
        # Show SSH info
        local ssh_info=$(get_instance_ssh_info "$instance_id")
        if [[ -n "$ssh_info" ]]; then
            local ssh_host=$(echo "$ssh_info" | cut -d: -f1)
            local ssh_port=$(echo "$ssh_info" | cut -d: -f2)
            echo ""
            echo "Connect with:"
            echo "  ssh -p $ssh_port root@$ssh_host"
        fi
        
        # Check for existing image
        local image_file="${backup_dir}/current_image.tar.gz"
        if [[ -f "$image_file" ]]; then
            echo ""
            log_info "Found existing Docker image backup"
            read -p "Load it to instance? [Y/n]: " load_image
            if [[ ! "$load_image" =~ ^[Nn]$ ]]; then
                cmd_load "$project"
            fi
        fi
    fi
}

# -----------------------------------------------------------------------------
# load <project>
# -----------------------------------------------------------------------------
cmd_load() {
    local project="$1"
    validate_project "$project"
    check_api_key
    
    local backup_dir="${PROJECT_ROOT}/projects/${project}__project-space/docker-backups"
    local image_file="${backup_dir}/current_image.tar.gz"
    local instance_id=$(get_instance_id "$project")
    
    if [[ -z "$instance_id" ]]; then
        log_error "No active instance for $project"
        echo "Run: ./vastai-manage.sh create $project"
        exit 1
    fi
    
    if [[ ! -f "$image_file" ]]; then
        log_error "No Docker image found: $image_file"
        exit 1
    fi
    
    local ssh_info=$(get_instance_ssh_info "$instance_id")
    if [[ -z "$ssh_info" ]]; then
        log_error "Cannot get SSH info for instance $instance_id"
        exit 1
    fi
    
    local ssh_host=$(echo "$ssh_info" | cut -d: -f1)
    local ssh_port=$(echo "$ssh_info" | cut -d: -f2)
    local ssh_key=$(get_ssh_key)
    local ssh_opts=""
    
    if [[ -n "$ssh_key" ]]; then
        ssh_opts="-i $ssh_key"
    fi
    
    local image_size=$(du -h "$image_file" | cut -f1)
    log_info "Uploading Docker image ($image_size) to instance..."
    
    # Upload image
    if ! retry_command "scp -P $ssh_port $ssh_opts -o StrictHostKeyChecking=no '$image_file' root@$ssh_host:/tmp/image.tar.gz"; then
        log_error "Upload failed"
        exit 1
    fi
    
    log_success "Upload complete"
    log_info "Loading Docker image..."
    
    # Load image on remote
    ssh -p "$ssh_port" $ssh_opts -o StrictHostKeyChecking=no "root@$ssh_host" \
        "docker load < /tmp/image.tar.gz && rm /tmp/image.tar.gz"
    
    log_success "Docker image loaded!"
    log_operation "$project" "LOAD" "image loaded to $instance_id"
    
    # Update timestamp
    date '+%Y-%m-%d %H:%M:%S' > "${backup_dir}/.last_loaded"
}

# -----------------------------------------------------------------------------
# save <project>
# -----------------------------------------------------------------------------
cmd_save() {
    local project="$1"
    validate_project "$project"
    check_api_key
    
    local backup_dir="${PROJECT_ROOT}/projects/${project}__project-space/docker-backups"
    local instance_id=$(get_instance_id "$project")
    local timestamp=$(date '+%Y-%m-%d_%H%M%S')
    
    if [[ -z "$instance_id" ]]; then
        log_error "No active instance for $project"
        exit 1
    fi
    
    local ssh_info=$(get_instance_ssh_info "$instance_id")
    if [[ -z "$ssh_info" ]]; then
        log_error "Cannot get SSH info for instance $instance_id"
        exit 1
    fi
    
    local ssh_host=$(echo "$ssh_info" | cut -d: -f1)
    local ssh_port=$(echo "$ssh_info" | cut -d: -f2)
    local ssh_key=$(get_ssh_key)
    local ssh_opts=""
    
    if [[ -n "$ssh_key" ]]; then
        ssh_opts="-i $ssh_key"
    fi
    
    mkdir -p "${backup_dir}/files"
    
    echo -e "${BLUE}=============================================="
    echo "  Saving Project: $project"
    echo -e "==============================================${NC}"
    echo ""
    
    # 1. File backup
    log_info "Creating file backup..."
    
    ssh -p "$ssh_port" $ssh_opts -o StrictHostKeyChecking=no "root@$ssh_host" \
        "cd /workspace && tar -czf /tmp/workspace_backup.tar.gz ."
    
    local file_backup="${backup_dir}/files/${timestamp}_session.tar.gz"
    
    if retry_command "scp -P $ssh_port $ssh_opts -o StrictHostKeyChecking=no root@$ssh_host:/tmp/workspace_backup.tar.gz '$file_backup'"; then
        log_success "File backup saved: $file_backup"
    else
        log_error "File backup failed"
        exit 1
    fi
    
    # 2. Docker image commit and save
    log_info "Committing Docker container..."
    
    ssh -p "$ssh_port" $ssh_opts -o StrictHostKeyChecking=no "root@$ssh_host" \
        "CONTAINER_ID=\$(docker ps -q | head -1) && \
         docker commit \$CONTAINER_ID ${project}:latest && \
         docker save ${project}:latest | gzip > /tmp/docker_image.tar.gz"
    
    # Rotate existing image
    local current_image="${backup_dir}/current_image.tar.gz"
    local previous_image="${backup_dir}/previous_image.tar.gz"
    
    if [[ -f "$current_image" ]]; then
        log_info "Rotating: current → previous"
        mv "$current_image" "$previous_image"
    fi
    
    # Download new image
    log_info "Downloading Docker image..."
    
    if retry_command "scp -P $ssh_port $ssh_opts -o StrictHostKeyChecking=no root@$ssh_host:/tmp/docker_image.tar.gz '$current_image'"; then
        local image_size=$(du -h "$current_image" | cut -f1)
        log_success "Docker image saved: $current_image ($image_size)"
    else
        log_error "Docker image download failed"
        # Restore previous
        if [[ -f "$previous_image" ]]; then
            mv "$previous_image" "$current_image"
        fi
        exit 1
    fi
    
    # Cleanup remote
    ssh -p "$ssh_port" $ssh_opts -o StrictHostKeyChecking=no "root@$ssh_host" \
        "rm -f /tmp/workspace_backup.tar.gz /tmp/docker_image.tar.gz" 2>/dev/null || true
    
    # Update timestamp
    date '+%Y-%m-%d %H:%M:%S' > "${backup_dir}/.last_saved"
    
    log_operation "$project" "SAVE" "files=$file_backup, image=$current_image"
    
    echo ""
    log_success "Save complete!"
    echo ""
    echo "Files backup: $file_backup"
    echo "Docker image: $current_image"
}

# -----------------------------------------------------------------------------
# save-files <project>
# -----------------------------------------------------------------------------
cmd_save_files() {
    local project="$1"
    validate_project "$project"
    check_api_key
    
    local backup_dir="${PROJECT_ROOT}/projects/${project}__project-space/docker-backups"
    local instance_id=$(get_instance_id "$project")
    local timestamp=$(date '+%Y-%m-%d_%H%M%S')
    
    if [[ -z "$instance_id" ]]; then
        log_error "No active instance for $project"
        exit 1
    fi
    
    local ssh_info=$(get_instance_ssh_info "$instance_id")
    if [[ -z "$ssh_info" ]]; then
        log_error "Cannot get SSH info for instance $instance_id"
        exit 1
    fi
    
    local ssh_host=$(echo "$ssh_info" | cut -d: -f1)
    local ssh_port=$(echo "$ssh_info" | cut -d: -f2)
    local ssh_key=$(get_ssh_key)
    local ssh_opts=""
    
    if [[ -n "$ssh_key" ]]; then
        ssh_opts="-i $ssh_key"
    fi
    
    mkdir -p "${backup_dir}/files"
    
    log_info "Creating quick file backup..."
    
    ssh -p "$ssh_port" $ssh_opts -o StrictHostKeyChecking=no "root@$ssh_host" \
        "cd /workspace && tar -czf /tmp/workspace_backup.tar.gz ."
    
    local file_backup="${backup_dir}/files/${timestamp}_session.tar.gz"
    
    if retry_command "scp -P $ssh_port $ssh_opts -o StrictHostKeyChecking=no root@$ssh_host:/tmp/workspace_backup.tar.gz '$file_backup'"; then
        local backup_size=$(du -h "$file_backup" | cut -f1)
        log_success "File backup saved: $file_backup ($backup_size)"
        
        ssh -p "$ssh_port" $ssh_opts -o StrictHostKeyChecking=no "root@$ssh_host" \
            "rm -f /tmp/workspace_backup.tar.gz" 2>/dev/null || true
        
        log_operation "$project" "SAVE-FILES" "$file_backup"
    else
        log_error "File backup failed"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# destroy <project>
# -----------------------------------------------------------------------------
cmd_destroy() {
    local project="$1"
    validate_project "$project"
    check_api_key
    
    local backup_dir="${PROJECT_ROOT}/projects/${project}__project-space/docker-backups"
    local instance_id=$(get_instance_id "$project")
    
    if [[ -z "$instance_id" ]]; then
        log_warn "No active instance for $project"
        return 0
    fi
    
    echo -e "${YELLOW}=============================================="
    echo "  Destroying Instance: $instance_id"
    echo "  Project: $project"
    echo -e "==============================================${NC}"
    echo ""
    
    # Safety check: backup recency
    local last_saved=""
    if [[ -f "${backup_dir}/.last_saved" ]]; then
        last_saved=$(cat "${backup_dir}/.last_saved")
    fi
    
    if [[ -z "$last_saved" ]]; then
        log_warn "No backup has ever been saved for this project!"
        read -p "Destroy anyway? You will LOSE ALL DATA! [y/N]: " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            log_info "Cancelled"
            exit 0
        fi
    else
        local today=$(date '+%Y-%m-%d')
        local saved_date=$(echo "$last_saved" | cut -d' ' -f1)
        
        if [[ "$saved_date" != "$today" ]]; then
            log_warn "Last backup was on $saved_date (not today)"
            read -p "Continue without saving? [y/N]: " confirm
            if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
                log_info "Run './vastai-manage.sh save $project' first"
                exit 0
            fi
        fi
    fi
    
    # Final confirmation
    read -p "Type project name to confirm destruction: " confirm_name
    if [[ "$confirm_name" != "$project" ]]; then
        log_error "Project name mismatch. Cancelled."
        exit 1
    fi
    
    log_info "Destroying instance $instance_id..."
    
    if vastai destroy instance "$instance_id"; then
        log_success "Instance destroyed"
        clear_instance_id "$project"
        log_operation "$project" "DESTROY" "instance_id=$instance_id"
    else
        log_error "Failed to destroy instance"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# archive <project> <milestone>
# -----------------------------------------------------------------------------
cmd_archive() {
    local project="$1"
    local milestone="$2"
    
    validate_project "$project"
    
    if [[ -z "$milestone" ]]; then
        log_error "Milestone name required"
        echo "Usage: ./vastai-manage.sh archive <project> <milestone>"
        exit 1
    fi
    
    local backup_dir="${PROJECT_ROOT}/projects/${project}__project-space/docker-backups"
    local archive_dir="${backup_dir}/archived"
    local current_image="${backup_dir}/current_image.tar.gz"
    
    if [[ ! -f "$current_image" ]]; then
        log_error "No current Docker image to archive"
        exit 1
    fi
    
    mkdir -p "$archive_dir"
    
    local archive_name="${milestone}_$(date '+%Y-%m-%d').tar.gz"
    local archive_path="${archive_dir}/${archive_name}"
    
    log_info "Archiving to: $archive_path"
    cp "$current_image" "$archive_path"
    
    local archive_size=$(du -h "$archive_path" | cut -f1)
    log_success "Archived: $archive_name ($archive_size)"
    log_operation "$project" "ARCHIVE" "$archive_name"
}

# -----------------------------------------------------------------------------
# cleanup <project> [--keep N]
# -----------------------------------------------------------------------------
cmd_cleanup() {
    local project="$1"
    local keep=15
    
    shift
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --keep)
                keep="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    validate_project "$project"
    
    local files_dir="${PROJECT_ROOT}/projects/${project}__project-space/docker-backups/files"
    
    if [[ ! -d "$files_dir" ]]; then
        log_info "No file backups to clean"
        return 0
    fi
    
    local count=$(find "$files_dir" -name "*.tar.gz" | wc -l | tr -d ' ')
    
    if [[ $count -le $keep ]]; then
        log_info "Only $count backups exist (keeping $keep). Nothing to clean."
        return 0
    fi
    
    local to_delete=$((count - keep))
    log_info "Found $count backups. Removing $to_delete oldest..."
    
    local deleted=0
    local freed=0
    
    find "$files_dir" -name "*.tar.gz" -type f -print0 | \
        xargs -0 ls -1t | tail -n "$to_delete" | while read file; do
        local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
        freed=$((freed + size))
        rm "$file"
        deleted=$((deleted + 1))
        echo "  Deleted: $(basename "$file")"
    done
    
    log_success "Cleanup complete"
    log_operation "$project" "CLEANUP" "removed $to_delete, kept $keep"
}

# -----------------------------------------------------------------------------
# status <project>
# -----------------------------------------------------------------------------
cmd_status() {
    local project="$1"
    validate_project "$project"
    
    local backup_dir="${PROJECT_ROOT}/projects/${project}__project-space/docker-backups"
    local instance_id=$(get_instance_id "$project")
    
    echo -e "${BOLD}Project: $project${NC}"
    echo "================================================"
    echo ""
    
    # Instance status
    echo -e "${CYAN}Instance:${NC}"
    if [[ -n "$instance_id" ]]; then
        local status=$(vastai show instance "$instance_id" --raw 2>/dev/null | jq -r '.actual_status // "unknown"')
        echo "  ID: $instance_id"
        echo "  Status: $status"
        
        if [[ "$status" == "running" ]]; then
            local ssh_info=$(get_instance_ssh_info "$instance_id")
            if [[ -n "$ssh_info" ]]; then
                local ssh_host=$(echo "$ssh_info" | cut -d: -f1)
                local ssh_port=$(echo "$ssh_info" | cut -d: -f2)
                echo "  SSH: ssh -p $ssh_port root@$ssh_host"
            fi
        fi
    else
        echo "  No active instance"
    fi
    echo ""
    
    # Docker image
    echo -e "${CYAN}Docker Image:${NC}"
    local current="${backup_dir}/current_image.tar.gz"
    local previous="${backup_dir}/previous_image.tar.gz"
    
    if [[ -f "$current" ]]; then
        local size=$(du -h "$current" | cut -f1)
        local mtime=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$current" 2>/dev/null || stat -c "%y" "$current" 2>/dev/null | cut -d. -f1)
        echo "  Current: $size (saved: $mtime)"
    else
        echo "  Current: None"
    fi
    
    if [[ -f "$previous" ]]; then
        local size=$(du -h "$previous" | cut -f1)
        echo "  Previous: $size"
    fi
    echo ""
    
    # File backups
    echo -e "${CYAN}File Backups:${NC}"
    local files_dir="${backup_dir}/files"
    if [[ -d "$files_dir" ]]; then
        local count=$(find "$files_dir" -name "*.tar.gz" | wc -l | tr -d ' ')
        local total_size=$(du -sh "$files_dir" 2>/dev/null | cut -f1 || echo "0")
        echo "  Count: $count"
        echo "  Total size: $total_size"
        
        if [[ $count -gt 0 ]]; then
            echo "  Latest:"
            find "$files_dir" -name "*.tar.gz" -type f -print0 | xargs -0 ls -1t | head -3 | while read f; do
                echo "    - $(basename "$f")"
            done
        fi
    else
        echo "  None"
    fi
    echo ""
    
    # Archives
    echo -e "${CYAN}Archives:${NC}"
    local archive_dir="${backup_dir}/archived"
    if [[ -d "$archive_dir" ]] && [[ -n "$(ls -A "$archive_dir" 2>/dev/null)" ]]; then
        ls -1 "$archive_dir" | head -5
    else
        echo "  None"
    fi
    echo ""
    
    # Timestamps
    echo -e "${CYAN}Timestamps:${NC}"
    if [[ -f "${backup_dir}/.last_saved" ]]; then
        echo "  Last saved: $(cat "${backup_dir}/.last_saved")"
    fi
    if [[ -f "${backup_dir}/.last_loaded" ]]; then
        echo "  Last loaded: $(cat "${backup_dir}/.last_loaded")"
    fi
}

# -----------------------------------------------------------------------------
# list-projects
# -----------------------------------------------------------------------------
list_projects() {
    echo -e "${BOLD}PINN-Experiments Projects${NC}"
    echo "=========================="
    echo ""
    
    local count=0
    for dir in "$PROJECT_ROOT"/projects/*__project-space; do
        if [[ -d "$dir" ]]; then
            local name=$(basename "$dir" | sed 's/__project-space$//')
            local backup_dir="$dir/docker-backups"
            local status=""
            
            # Check for instance
            if [[ -f "${backup_dir}/.current_instance_id" ]]; then
                status="${GREEN}●${NC} Active"
            elif [[ -f "${backup_dir}/current_image.tar.gz" ]]; then
                status="${BLUE}●${NC} Has backup"
            else
                status="${YELLOW}○${NC} No backup"
            fi
            
            printf "  %-35s %b\n" "$name" "$status"
            count=$((count + 1))
        fi
    done
    
    echo ""
    echo "Total: $count projects"
}

# -----------------------------------------------------------------------------
# list-instances
# -----------------------------------------------------------------------------
cmd_list_instances() {
    check_api_key
    
    echo -e "${BOLD}Running Vast.ai Instances${NC}"
    echo "========================="
    echo ""
    
    vastai show instances
    
    echo ""
    echo -e "${CYAN}Project Instance Mapping:${NC}"
    
    for dir in "$PROJECT_ROOT"/projects/*__project-space; do
        if [[ -d "$dir" ]]; then
            local name=$(basename "$dir" | sed 's/__project-space$//')
            local id_file="$dir/docker-backups/.current_instance_id"
            
            if [[ -f "$id_file" ]]; then
                local instance_id=$(cat "$id_file")
                echo "  $name → $instance_id"
            fi
        fi
    done
}

# -----------------------------------------------------------------------------
# emergency-restore <project>
# -----------------------------------------------------------------------------
cmd_emergency_restore() {
    local project="$1"
    validate_project "$project"
    check_api_key
    
    local backup_dir="${PROJECT_ROOT}/projects/${project}__project-space/docker-backups"
    local files_dir="${backup_dir}/files"
    local instance_id=$(get_instance_id "$project")
    
    if [[ -z "$instance_id" ]]; then
        log_error "No active instance for $project"
        echo "Create one first: ./vastai-manage.sh create $project"
        exit 1
    fi
    
    if [[ ! -d "$files_dir" ]] || [[ -z "$(ls -A "$files_dir" 2>/dev/null)" ]]; then
        log_error "No file backups found"
        exit 1
    fi
    
    echo -e "${YELLOW}=============================================="
    echo "  Emergency Restore: $project"
    echo -e "==============================================${NC}"
    echo ""
    
    echo "Available backups:"
    local i=1
    local backups=()
    while IFS= read -r file; do
        local size=$(du -h "$file" | cut -f1)
        echo "  $i) $(basename "$file") ($size)"
        backups+=("$file")
        i=$((i + 1))
    done < <(find "$files_dir" -name "*.tar.gz" -type f -print0 | xargs -0 ls -1t)
    
    echo ""
    read -p "Select backup number [1]: " selection
    selection=${selection:-1}
    
    local selected_backup="${backups[$((selection - 1))]}"
    if [[ -z "$selected_backup" || ! -f "$selected_backup" ]]; then
        log_error "Invalid selection"
        exit 1
    fi
    
    local ssh_info=$(get_instance_ssh_info "$instance_id")
    local ssh_host=$(echo "$ssh_info" | cut -d: -f1)
    local ssh_port=$(echo "$ssh_info" | cut -d: -f2)
    local ssh_key=$(get_ssh_key)
    local ssh_opts=""
    
    if [[ -n "$ssh_key" ]]; then
        ssh_opts="-i $ssh_key"
    fi
    
    log_info "Uploading backup..."
    scp -P "$ssh_port" $ssh_opts -o StrictHostKeyChecking=no "$selected_backup" "root@$ssh_host:/tmp/restore.tar.gz"
    
    log_info "Extracting to /workspace..."
    ssh -p "$ssh_port" $ssh_opts -o StrictHostKeyChecking=no "root@$ssh_host" \
        "cd /workspace && tar -xzf /tmp/restore.tar.gz && rm /tmp/restore.tar.gz"
    
    log_success "Restore complete!"
    log_operation "$project" "EMERGENCY-RESTORE" "$(basename "$selected_backup")"
}

# -----------------------------------------------------------------------------
# dashboard
# -----------------------------------------------------------------------------
cmd_dashboard() {
    check_api_key
    
    echo -e "${BOLD}PINN-Experiments Project Dashboard${NC}"
    echo "====================================="
    echo ""
    
    printf "%-28s | %-12s | %-10s | %-12s | %s\n" "Project" "Last Backup" "Instance" "Docker Image" "Status"
    printf "%-28s-+-%-12s-+-%-10s-+-%-12s-+-%s\n" "----------------------------" "------------" "----------" "------------" "--------"
    
    local total_size=0
    local projects_with_backups=0
    local active_instances=0
    local total_projects=0
    
    for dir in "$PROJECT_ROOT"/projects/*__project-space; do
        if [[ -d "$dir" ]]; then
            local name=$(basename "$dir" | sed 's/__project-space$//')
            local backup_dir="$dir/docker-backups"
            
            # Truncate long names
            local display_name="$name"
            if [[ ${#name} -gt 26 ]]; then
                display_name="${name:0:23}..."
            fi
            
            # Last backup date
            local last_backup="Never"
            if [[ -f "${backup_dir}/.last_saved" ]]; then
                last_backup=$(cat "${backup_dir}/.last_saved" | cut -d' ' -f1)
                projects_with_backups=$((projects_with_backups + 1))
            fi
            
            # Instance
            local instance="None"
            if [[ -f "${backup_dir}/.current_instance_id" ]]; then
                instance=$(cat "${backup_dir}/.current_instance_id")
                active_instances=$((active_instances + 1))
            fi
            
            # Docker image size
            local image_size="-"
            local current="${backup_dir}/current_image.tar.gz"
            if [[ -f "$current" ]]; then
                image_size=$(du -h "$current" | cut -f1)
                local bytes=$(stat -f%z "$current" 2>/dev/null || stat -c%s "$current" 2>/dev/null || echo 0)
                total_size=$((total_size + bytes))
            fi
            
            # Status
            local status=""
            if [[ -n "$(ls -A "${backup_dir}/files" 2>/dev/null)" && -f "$current" ]]; then
                status="${GREEN}✓ Ready${NC}"
            elif [[ -n "$(ls -A "${backup_dir}/files" 2>/dev/null)" ]]; then
                status="${YELLOW}Files only${NC}"
            elif [[ -f "$current" ]]; then
                status="${YELLOW}Image only${NC}"
            else
                status="${RED}No backups${NC}"
            fi
            
            printf "%-28s | %-12s | %-10s | %-12s | %b\n" "$display_name" "$last_backup" "$instance" "$image_size" "$status"
            total_projects=$((total_projects + 1))
        fi
    done
    
    echo ""
    echo "Active Instances: $active_instances"
    
    # Format total size
    if [[ $total_size -ge 1073741824 ]]; then
        echo "Total Backup Size: $(echo "scale=1; $total_size / 1073741824" | bc) GB"
    else
        echo "Total Backup Size: $(echo "scale=1; $total_size / 1048576" | bc) MB"
    fi
    
    echo "Projects with backups: $projects_with_backups/$total_projects"
}

# -----------------------------------------------------------------------------
# configure-keys
# -----------------------------------------------------------------------------
cmd_configure_keys() {
    local script="${PROJECT_ROOT}/configure-keys.sh"
    
    if [[ -f "$script" ]]; then
        exec "$script"
    else
        log_error "configure-keys.sh not found"
        exit 1
    fi
}

# =============================================================================
# MAIN
# =============================================================================

cd "$PROJECT_ROOT" 2>/dev/null || true

case "${1:-}" in
    create)
        shift
        cmd_create "$@"
        ;;
    load)
        cmd_load "$2"
        ;;
    save)
        cmd_save "$2"
        ;;
    save-files)
        cmd_save_files "$2"
        ;;
    destroy)
        cmd_destroy "$2"
        ;;
    archive)
        cmd_archive "$2" "$3"
        ;;
    cleanup)
        shift
        cmd_cleanup "$@"
        ;;
    status)
        cmd_status "$2"
        ;;
    list-projects|projects)
        list_projects
        ;;
    list-instances|list)
        cmd_list_instances
        ;;
    emergency-restore)
        cmd_emergency_restore "$2"
        ;;
    dashboard)
        cmd_dashboard
        ;;
    configure-keys)
        cmd_configure_keys
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac
