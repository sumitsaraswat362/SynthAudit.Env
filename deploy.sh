#!/bin/bash
# SynthAudit.Env — Deployment Script
# Run from the SynthAudit.Env directory

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  SynthAudit.Env — Deployment                           ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ─── Step 1: GitHub ─────────────────────────────────────────
echo ""
echo "Step 1: Push to GitHub"
echo "  First create repo at: https://github.com/new"
echo "  Name: SynthAudit.Env"
echo "  Visibility: Public"
echo ""
read -p "  GitHub repo URL (e.g. https://github.com/sumitsaraswat/SynthAudit.Env.git): " GITHUB_URL

if [ -n "$GITHUB_URL" ]; then
    git remote remove origin 2>/dev/null || true
    git remote add origin "$GITHUB_URL"
    git branch -M main
    git push -u origin main
    echo "✓ Pushed to GitHub"
fi

# ─── Step 2: HuggingFace Space ──────────────────────────────
echo ""
echo "Step 2: Deploy to HuggingFace Spaces"
echo "  First create space at: https://huggingface.co/new-space"
echo "  Name: SynthAudit-Env"
echo "  SDK: Docker"
echo "  Hardware: CPU Basic (free)"
echo ""
read -p "  HuggingFace space URL (e.g. https://huggingface.co/spaces/sumitsaraswat/SynthAudit-Env): " HF_URL

if [ -n "$HF_URL" ]; then
    # Convert space URL to git URL
    HF_GIT="https://huggingface.co/spaces/$(echo $HF_URL | sed 's|.*/spaces/||')"
    git remote remove hf 2>/dev/null || true
    git remote add hf "$HF_GIT"
    git push hf main
    echo "✓ Pushed to HuggingFace Spaces"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Deployment complete!                                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
