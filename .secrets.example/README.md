# Secrets Examples

This folder contains templates showing what secrets are needed.

Copy the structure to `.secrets/` and fill in your actual values:

## Update (April 17, 2026)

- Secret templates remain the tracked source of truth; real values should stay in `.secrets/` or `private/`.
- The repo now has more explicit key-management and sync docs, so this folder sits inside a larger credential workflow.
- Keep public-safe placeholders here only; do not mirror real tokens or machine addresses back into git.

```bash
cp .env.example ../.secrets/.env
# Then edit ../.secrets/.env with your real keys
```

See `KEYS.md` in the project root for full instructions.
