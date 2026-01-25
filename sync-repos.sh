#!/bin/bash
# Sync script: Pushes to private and syncs filtered version to public
# Usage: ./sync-repos.sh "commit message"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_DIR="/tmp/PINN-Experiments-public"
SECRETS_FILE="$SCRIPT_DIR/.secrets-filter.txt"
COMMIT_MSG="${1:-Sync updates}"

echo "=== Pushing to PRIVATE repository ==="
cd "$SCRIPT_DIR"
git push origin main

echo ""
echo "=== Syncing to PUBLIC repository ==="

# Ensure public directory exists
if [ ! -d "$PUBLIC_DIR/.git" ]; then
    echo "Cloning public repo..."
    rm -rf "$PUBLIC_DIR"
    git clone https://github.com/lekandigital/PINN-Experiments-public.git "$PUBLIC_DIR"
fi

# Sync files (exclude .git, .claude, research-docs, and secrets)
echo "Syncing files..."
rsync -av --delete \
    --exclude='.git' \
    --exclude='.claude' \
    --exclude='.secrets-filter.txt' \
    --exclude='research-docs' \
    "$SCRIPT_DIR/" "$PUBLIC_DIR/"

# Replace secrets in public copy using the config file
echo "Filtering secrets..."
cd "$PUBLIC_DIR"
if [ -f "$SECRETS_FILE" ]; then
    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue

        # Parse SECRET==>REPLACEMENT format
        secret="${line%%==>*}"
        replacement="${line##*==>}"

        # Find and replace in text files
        find . -type f \( -name "*.md" -o -name "*.sh" -o -name "*.yaml" -o -name "*.yml" -o -name "*.txt" -o -name "*.json" -o -name "*.py" \) 2>/dev/null | while read -r file; do
            if grep -q "$secret" "$file" 2>/dev/null; then
                # Use sed with different delimiter to handle special chars
                sed -i '' "s|$secret|$replacement|g" "$file" 2>/dev/null || true
            fi
        done
    done < "$SECRETS_FILE"
fi

# Commit and push to public
echo "Committing to public..."
git add -A
if git diff --staged --quiet; then
    echo "No changes to commit to public repo"
else
    git commit -m "$COMMIT_MSG

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
    git push origin main
fi

echo ""
echo "=== Successfully synced to both repositories ==="
echo "Private: https://github.com/lekandigital/PINN-Experiments-private"
echo "Public:  https://github.com/lekandigital/PINN-Experiments-public"
