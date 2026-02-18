#!/bin/bash
# Sync files from upstream main -> local main -> tobias branch
# Complete workflow: GitHub fork update -> local main update -> tobias branch sync

# RUN: ./local/sync_ovos_from_main.sh



set -e  # Exit on error

echo "=========================================="
echo "  Syncing from Upstream to Tobias Branch"
echo "=========================================="
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Store current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: ${CURRENT_BRANCH}"
echo ""

# Step 1: Fetch from upstream (the original repo you forked from)
echo "Step 1: Fetching from upstream..."
if git remote get-url upstream &>/dev/null; then
    git fetch upstream
    echo "   ✓ Fetched from upstream"
else
    echo "   ⚠ No upstream remote found, skipping..."
fi

# Step 2: Update local main branch
echo ""
echo "Step 2: Updating local main branch..."
git checkout main
git pull --ff-only upstream main 2>/dev/null || git pull --ff-only origin main || git pull --rebase origin main
echo "   ✓ Local main updated"

# Step 3: Push updated main to your fork (origin)
echo ""
echo "Step 3: Pushing to origin/main (your fork)..."
git push origin main
echo "   ✓ Fork updated on GitHub"

# Step 4: Switch back to tobias branch
echo ""
echo "Step 4: Switching to ${CURRENT_BRANCH} branch..."
git checkout "${CURRENT_BRANCH}"

# Step 5: Backup and sync files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo ""
echo "Step 5: Syncing files from main to ${CURRENT_BRANCH}..."

# Backup ovos.py if it exists
if [ -f ovos.py ]; then
# Create backup directory if it does not exist
mkdir -p backup

    OVOS_BACKUP="backup/ovos_tobias_backup_${TIMESTAMP}.py"
    cp ovos.py "${OVOS_BACKUP}"
    echo "   • Backed up ovos.py -> ${OVOS_BACKUP}"
fi

# Backup README.md if it exists
if [ -f README.md ]; then
    README_BACKUP="backup/README_tobias_backup_${TIMESTAMP}.md"
    cp README.md "${README_BACKUP}"
    echo "   • Backed up README.md -> ${README_BACKUP}"
fi

# Sync files from main
echo ""
echo "   Syncing from main branch..."
git checkout main -- ovos.py README.md 2>/dev/null || echo "   (some files may not exist in main)"

# Stage changes
git add ovos.py README.md 2>/dev/null || true

echo ""
echo "=========================================="
echo "  ✅ Sync Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  • Updated local main from upstream"
echo "  • Pushed to your GitHub fork"
echo "  • Synced ovos.py and README.md to ${CURRENT_BRANCH}"
if [ -f "${OVOS_BACKUP}" ]; then
    echo "  • Backup: ${OVOS_BACKUP}"
fi
if [ -f "${README_BACKUP}" ]; then
    echo "  • Backup: ${README_BACKUP}"
fi
echo ""
echo "Next steps:"
echo "  • Review: git diff --staged"
echo "  • Commit: git commit -m 'Sync ovos.py and README.md from main'"
echo "  • Or unstage: git restore --staged ovos.py README.md"
echo ""
