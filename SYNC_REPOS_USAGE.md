# Sync-Repos Usage Guide

## What is sync-repos.sh?

`sync-repos.sh` is a script that automatically pushes your commits to **two separate GitHub repositories**:

1. **PINN-Experiments-private** (this repo): Full content with all credentials
2. **PINN-Experiments-public**: Clean version with secrets filtered and private files excluded

This allows you to:
- Keep a private working copy with all credentials and research notes
- Maintain a public-facing clean version for sharing and collaboration
- Automatically filter sensitive information from the public version

## Quick Usage

### Step 1: Make Your Changes and Commit

```bash
# Edit files as normal
git add .
git commit -m "Your descriptive commit message"
```

### Step 2: Sync to Both Repositories

```bash
./sync-repos.sh "Your descriptive commit message"
```

That's it! The script will:
1. Push to **private** repo (origin) with full content
2. Sync to **public** repo with secrets filtered and research-docs excluded

## What Gets Filtered?

### Secrets Automatically Replaced

Defined in `.secrets-filter.txt`:

| Secret | Replacement |
|--------|------------|
| `REDACTED_PASSWORD` | `REDACTED_PASSWORD` |
| `REDACTED_SERVER` | `REDACTED_SERVER` |
| `REDACTED_API_KEY` | `REDACTED_API_KEY` |
| `REDACTED_VASTAI_API_KEY` | `REDACTED_VASTAI_API_KEY` |

### Directories Never Synced to Public

- `research-docs/` - Private research documentation
- `.claude/` - IDE settings cache
- `.git/` - Each repo maintains separate history
- `.secrets-filter.txt` - Credentials config file

## Adding New Secrets to Filter

If you commit a new credential by accident:

1. **Add it to `.secrets-filter.txt`:**
   ```
   your_new_secret==>REDACTED_VALUE
   ```

2. **Run sync:**
   ```bash
   ./sync-repos.sh "Filter new credential"
   ```

3. **The public repo will be updated** with the secret replaced in current files

> **Note**: Existing git history in the public repo is not retroactively cleaned. Use `./sync-repos.sh` immediately after committing to prevent old commits from lingering.

## Common Workflows

### Push a new feature

```bash
# Develop your feature
git add .
git commit -m "Add new PINN model architecture"

# Sync to both repos
./sync-repos.sh "Add new PINN model architecture"
```

### Update research notes (stays private)

```bash
# Add to research-docs (won't sync to public)
git add research-docs/new_findings.txt
git commit -m "Add research notes on convergence patterns"

# Still sync (research-docs will be excluded)
./sync-repos.sh "Add research notes on convergence patterns"
```

### Fix a bug

```bash
git add src/
git commit -m "Fix: incorrect boundary condition in solver"

./sync-repos.sh "Fix: incorrect boundary condition in solver"
```

## Understanding the Script Flow

When you run `./sync-repos.sh "message"`:

```
Step 1: Push to Private
  └─> git push origin main
      └─> Your commit goes to PINN-Experiments-private (full content)

Step 2: Sync Files
  └─> rsync /Users/lekan/Dev/PINN-Experiments → /tmp/PINN-Experiments-public
      └─> Copies files EXCEPT: .git, research-docs, .claude, .secrets-filter.txt

Step 3: Filter Secrets
  └─> Replace all credentials in public copy
      └─> Uses patterns from .secrets-filter.txt
      └─> Scans: *.md, *.sh, *.yaml, *.yml, *.txt, *.json, *.py

Step 4: Commit & Push to Public
  └─> git commit -m "Your message"
      └─> git push origin main
      └─> Public version goes to PINN-Experiments-public (filtered content)
```

## Repository Structure

```
Your Machine
├── /Users/lekan/Dev/PINN-Experiments/
│   ├── (Working directory - full content)
│   ├── .secrets-filter.txt (gitignored)
│   └── sync-repos.sh
│
GitHub
├── PINN-Experiments-private
│   └── (Private repo - full history & content)
│
└── PINN-Experiments-public
    └── (Public repo - filtered content, clean history)
```

## Important Rules

✅ **DO:**
- Use `./sync-repos.sh` after every commit
- Add new credentials to `.secrets-filter.txt` immediately
- Keep `.secrets-filter.txt` in `.gitignore` (already configured)
- Use the private repo as your main working copy

❌ **DON'T:**
- Push directly to the public repo
- Forget to run sync after committing
- Leave secrets unfiltered (run sync immediately if you do)
- Edit the public repo directly

## Troubleshooting

### Script Fails on First Run

Make sure you're in the repo directory:
```bash
cd /Users/lekan/Dev/PINN-Experiments
./sync-repos.sh "message"
```

### Public Repo Shows Old Secrets

The script filters the **current** files. If secrets appear in old commits:
1. Public repo is periodically recreated fresh
2. Or use the `git-filter-repo` method for deep historical cleaning
3. Contact admin if historical secrets are visible

### Changes Not Appearing in Public

Run sync manually:
```bash
./sync-repos.sh "Sync latest changes"
```

### Permission Errors

Ensure script is executable:
```bash
chmod +x sync-repos.sh
```

## Git Alias Alternative

You can also use the configured git alias:

```bash
git sync-push
```

This runs: `git push origin main && ./sync-repos.sh "$(git log -1 --format=%s)"`

But the explicit `./sync-repos.sh` method is recommended for clarity.

## Files Related to Sync

| File | Purpose | Status |
|------|---------|--------|
| `sync-repos.sh` | Main sync script | Committed (in both repos) |
| `.secrets-filter.txt` | Credentials to filter | Gitignored (private only) |
| `.gitignore` | Includes .secrets-filter.txt | Committed (both repos) |
| `GIT_COMMIT_SYNCING_INSTRUCTIONS.md` | User guide | Committed (both repos) |
| `SYNC_REPOS_USAGE.md` | This file (private reference) | Committed (private only) |

## When to Sync

- **After every commit** - ensures both repos stay in sync
- **Immediately after adding credentials** - prevents them from lingering
- **Before sharing work** - ensures public version is current
- **At end of day** - good checkpoint

## Questions?

Refer to:
- `GIT_COMMIT_SYNCING_INSTRUCTIONS.md` - General workflow guide
- `SYNC_REPOS_USAGE.md` - This file
- `sync-repos.sh` - Actual script implementation
