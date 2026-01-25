# PINN-Experiments: API Keys & Secrets Management

This document describes how to manage API keys and secrets for the PINN-Experiments project.

---

## Principles

1. **Never commit secrets to git** — All key files are gitignored
2. **One location** — All project keys live in `.secrets/`
3. **Documented** — Every required key has setup instructions
4. **Separated** — Each service gets its own file

---

## Folder Structure

```
~/Dev/PINN-Experiments/
├── .secrets/                          # GITIGNORED - All secrets here
│   ├── README.md                      # Setup instructions (safe to keep)
│   ├── vast_api_key                   # Vast.ai API key
│   ├── hf_token                       # HuggingFace token (if needed)
│   ├── wandb_api_key                  # Weights & Biases (if needed)
│   └── .env                           # Combined env file for scripts
│
├── .secrets.example/                  # TRACKED - Template showing required keys
│   ├── README.md
│   ├── vast_api_key.example
│   ├── hf_token.example
│   ├── wandb_api_key.example
│   └── .env.example
│
└── .gitignore                         # Includes .secrets/
```

---

## Required Keys

### Vast.ai API Key

**Purpose**: Authenticate with Vast.ai CLI for instance management

**How to get it**:
1. Go to https://cloud.vast.ai/
2. Log in to your account
3. Click on your account menu (top right)
4. Select "Account"
5. Find "API Keys" section
6. Copy your API key (or generate a new one)

### Automated Setup (Recommended)

Run the interactive configuration script:

```bash
cd ~/Dev/PINN-Experiments
./configure-keys.sh
```

Or use vastai-manage.sh:

```bash
./vastai-manage.sh configure-keys
```

This handles all key setup, permissions, and verification automatically.

### Manual Setup
```bash
# Save to project secrets
echo "YOUR_API_KEY_HERE" > ~/Dev/PINN-Experiments/.secrets/vast_api_key

# Also configure Vast.ai CLI (uses its own location)
echo "YOUR_API_KEY_HERE" > ~/.vast_api_key

# Verify
vastai show user
```

**Used by**: `vastai-manage.sh`, Vast.ai CLI

---

### HuggingFace Token (Optional)

**Purpose**: Download gated models/datasets from HuggingFace

**How to get it**:
1. Go to https://huggingface.co/settings/tokens
2. Create new token with "Read" access
3. Copy the token

**Setup**:
```bash
echo "hf_YOUR_TOKEN_HERE" > ~/Dev/PINN-Experiments/.secrets/hf_token
```

**Used by**: Training scripts that download HF models

---

### Weights & Biases API Key (Optional)

**Purpose**: Log experiments to W&B dashboard

**How to get it**:
1. Go to https://wandb.ai/settings
2. Find "API keys" section
3. Copy your key

**Setup**:
```bash
echo "YOUR_WANDB_KEY" > ~/Dev/PINN-Experiments/.secrets/wandb_api_key
```

**Used by**: Training scripts with W&B logging

---

## Using Keys in Scripts

### Option 1: Source the .env file

```bash
# In your script or shell
source ~/Dev/PINN-Experiments/.secrets/.env
echo $VAST_API_KEY
```

### Option 2: Read individual key files

```bash
# In bash scripts
VAST_API_KEY=$(cat ~/Dev/PINN-Experiments/.secrets/vast_api_key)
```

### Option 3: Export to environment

```bash
# Add to ~/.zshrc or ~/.bashrc for persistent access
export VAST_API_KEY=$(cat ~/Dev/PINN-Experiments/.secrets/vast_api_key 2>/dev/null)
```

---

## The .env File Format

The `.secrets/.env` file combines all keys for easy sourcing:

```bash
# Vast.ai
VAST_API_KEY=your_vast_key_here

# HuggingFace (optional)
HF_TOKEN=hf_your_token_here

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_key_here

# Add other project-specific vars as needed
```

---

## Security Checklist

- [ ] `.secrets/` is in `.gitignore`
- [ ] No keys in any committed files
- [ ] Keys have minimal required permissions
- [ ] Keys are rotated periodically (every 6-12 months)
- [ ] Old keys are revoked after rotation

---

## Rotating Keys

When you need to rotate a key:

1. Generate new key in the service's dashboard
2. Update the file in `.secrets/`
3. Update `~/.vast_api_key` if it's the Vast.ai key
4. Test that the new key works: `vastai show user`
5. Revoke the old key in the service's dashboard

---

## Recovering from Key Exposure

If a key is accidentally committed or exposed:

1. **Immediately revoke** the exposed key in the service's dashboard
2. Generate a new key
3. Update `.secrets/` and any other locations
4. Check git history — you may need to:
   ```bash
   # Remove from history (nuclear option)
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch .secrets/*" \
     --prune-empty --tag-name-filter cat -- --all
   ```
5. Force push if the repo is remote (coordinate with collaborators)

---

## Adding New Keys

1. Create the key file in `.secrets/`:
   ```bash
   echo "NEW_KEY_VALUE" > ~/Dev/PINN-Experiments/.secrets/new_service_key
   ```

2. Add an example file in `.secrets.example/`:
   ```bash
   echo "# Get this from https://service.com/settings" > ~/Dev/PINN-Experiments/.secrets.example/new_service_key.example
   ```

3. Add to `.secrets/.env`:
   ```bash
   echo "NEW_SERVICE_KEY=your_key_here" >> ~/Dev/PINN-Experiments/.secrets/.env
   ```

4. Document in this file (KEYS.md)

---

## Troubleshooting

### "Permission denied" from Vast.ai

- Check key exists: `cat ~/.vast_api_key`
- Verify key is valid: `vastai show user`
- Regenerate key if needed

### Key file has wrong permissions

```bash
# Make key files readable only by you
chmod 600 ~/Dev/PINN-Experiments/.secrets/*
```

### Accidentally committed a key

See "Recovering from Key Exposure" above.

---

## Related Files

- `.gitignore` — Must include `.secrets/`
- `vastai-manage.sh` — Uses Vast.ai API key
- `WORKFLOW.md` — References this document
