> **Note**: This prompt has been incorporated into `CLAUDE_CODE_MONOREPO_SETUP.md`. 
> Keep this file for reference but use the consolidated prompt for execution.

---

# Prompt: Improve WORKFLOW.md for Vast.ai GPU Rental Workflow

## Context

I have a `WORKFLOW.md` file that documents my Vast.ai GPU rental workflow for the PINN-Experiments monorepo. The repo contains 18+ projects across four categories (PINN, GNN, NIF, Hybrid).

The workflow covers:
- Instance lifecycle (create → load → work → save → destroy)
- Two backup types (Docker image + file backup, no exclusions)
- Folder structure and git integration
- The `vastai-manage.sh` script reference
- Artifact preservation policy (regenerable vs costly artifacts)
- Pre-save checklist and verification procedures
- Emergency procedures and troubleshooting
- Source of truth hierarchy (git > docker image > file backup > instance)

The workflow is designed for cost savings: only pay for GPU time when actively working, save complete backups locally, destroy instances when done.

## Current Document Location

`~/Dev/PINN-Experiments/WORKFLOW.md`

## Current Project Structure

```
~/Dev/PINN-Experiments/
├── .git/
├── .gitignore
├── README.md
├── WORKFLOW.md
├── vastai-manage.sh
├── research-docs/                    # Centralized research documentation
│   ├── ChatGPT-*.md.txt
│   └── parts1-originals/
└── projects/
    ├── [18+ project-space folders]
    └── each containing:
        ├── README.md
        ├── <project-name>/           # Source code
        ├── docker/                   # Dockerfile, compose
        ├── docs/                     # Project-specific docs
        └── docker-backups/           # IGNORED - backups here
```

## What I Want You To Do

Review and improve the WORKFLOW.md document. Consider:

### Clarity & Completeness

1. Are there any steps in the workflow that are unclear or missing?
2. Does the folder structure diagram accurately represent the actual layout?
3. Are the script commands and their purposes well explained?
4. Is the relationship between Docker images and file backups clear?
5. Is the Source of Truth Hierarchy clear and actionable?

### Technical Accuracy

1. Are the Vast.ai CLI commands accurate?
2. Are the Docker commands (commit, save, load) correctly described?
3. Are the backup file paths consistent throughout the document?
4. Does the architecture diagram accurately show the flow?
5. Are the .gitignore patterns correct and complete?

### Practical Improvements

1. Add any missing edge cases to emergency procedures
2. Suggest better backup rotation strategies if applicable
3. Identify any race conditions or failure modes not covered
4. Add timing estimates where helpful (upload/download times, etc.)
5. Is the artifact preservation policy practical and enforceable?

### Structure & Formatting

1. Is the table of contents complete?
2. Are the sections in logical order?
3. Would any sections benefit from additional diagrams?
4. Are code blocks and command examples clear?

### Integration Points

1. Does the git integration section fully cover what should/shouldn't be tracked?
2. Is the .gitignore example complete for this use case?
3. Are there any missing files that should be mentioned in "Related Files"?
4. Is the research-docs/ folder properly documented?

## Specific Questions to Address

1. **First-time setup**: Is it clear what a new user needs to do before this workflow works? (Vast.ai CLI setup, SSH keys, Docker installation, etc.)

2. **Multi-project handling**: How should someone manage multiple projects running simultaneously on different instances?

3. **Partial saves**: The `save-files` command for quick mid-session backups - is this documented well enough?

4. **Automation**: Could any parts of this workflow be automated further? (cron jobs, pre-commit hooks, etc.)

5. **Cost tracking**: Should there be a section on monitoring/tracking Vast.ai costs?

6. **Project categories**: Are the PINN/GNN/NIF/Hybrid categories useful? Should they be documented differently?

7. **REGENERATE.md**: Is the convention for documenting regenerable artifacts clear?

8. **Pre-save checklist**: Should this be integrated into the script as prompts?

## Output Format

Please provide:
1. A summary of suggested improvements
2. Specific edits with before/after where applicable
3. Any new sections you recommend adding
4. Any sections you recommend removing or consolidating
5. Any inconsistencies or errors found

## Additional Context

- This document lives in a git repo and is tracked
- Target audience: myself and potentially collaborators on PINN research
- The `vastai-manage.sh` script is generated from a separate prompt and handles the mechanics
- I use this on a Mac with Docker CLI installed locally
- Docker containers run INSIDE the Vast.ai instances (not Vast.ai's Docker rental mode)
- File backups have NO exclusions - we capture everything including weights and data
- The research-docs/ folder contains ChatGPT conversation exports and research plans
