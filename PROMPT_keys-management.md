> **Note**: This prompt has been incorporated into `CLAUDE_CODE_MONOREPO_SETUP.md`. 
> Keep this file for reference but use the consolidated prompt for execution.

---

# Prompt: Manage API Keys for PINN-Experiments

## Context

I have a secrets management system for my PINN-Experiments project:

```
~/Dev/PINN-Experiments/
├── .secrets/                    # GITIGNORED - Actual secrets
│   ├── README.md
│   ├── vast_api_key
│   ├── hf_token
│   ├── wandb_api_key
│   └── .env
│
├── .secrets.example/            # TRACKED - Templates
│   ├── README.md
│   ├── vast_api_key.example
│   ├── hf_token.example
│   ├── wandb_api_key.example
│   └── .env.example
│
├── KEYS.md                      # Documentation
└── .gitignore                   # Includes .secrets/
```

## Current Keys

| Key | Service | Purpose | Status |
|-----|---------|---------|--------|
| `vast_api_key` | Vast.ai | GPU instance management | Required |
| `hf_token` | HuggingFace | Model/dataset downloads | Optional |
| `wandb_api_key` | Weights & Biases | Experiment tracking | Optional |

## What I Need Help With

[Choose one or describe your task]

### Option A: Add a New Key

I need to add a new API key for: `[SERVICE_NAME]`

Please:
1. Create the key file in `.secrets/`
2. Create an example file in `.secrets.example/`
3. Add to `.secrets/.env`
4. Update KEYS.md with setup instructions

### Option B: Rotate a Key

I need to rotate my `[KEY_NAME]` key.

Please provide:
1. Steps to generate new key in the service dashboard
2. Commands to update all locations
3. Verification commands
4. Reminder to revoke old key

### Option C: Troubleshoot Key Issues

I'm getting this error: `[ERROR_MESSAGE]`

When running: `[COMMAND]`

Please help diagnose and fix.

### Option D: Add Key to a Script

I need to use the `[KEY_NAME]` key in my script at `[SCRIPT_PATH]`.

Please show how to:
1. Load the key securely
2. Use it in the script
3. Handle missing key gracefully

### Option E: Set Up Keys on New Machine

I'm setting up a new development machine and need to:
1. Copy my secrets structure
2. Set up all required keys
3. Verify everything works

---

## Security Reminders

- Never commit `.secrets/` to git
- Never echo keys in logs or output
- Use `chmod 600` for key files
- Rotate keys if exposed
