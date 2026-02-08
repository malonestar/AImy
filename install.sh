#!/bin/bash
set -e

# -----------------------------
# Resolve project root
# -----------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME="aimy_venv"
PY="$PROJECT_ROOT/$VENV_NAME/bin/python"
REQS_FILE="$PROJECT_ROOT/requirements.txt"
BASHRC_FILE="$HOME/.bashrc"
ALIAS_NAME="aimyenv"
ACTIVATE_SCRIPT_PATH="$PROJECT_ROOT/$VENV_NAME/bin/activate"

echo "=============================================="
echo "AImy Installer"
echo "Project root: $PROJECT_ROOT"
echo "=============================================="

# -----------------------------
# System dependencies
# -----------------------------
echo "[SYSTEM] Installing Picamera2 + OpenCV deps..."
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv libopencv-dev

# -----------------------------
# Create venv
# -----------------------------
if [ ! -d "$PROJECT_ROOT/$VENV_NAME" ]; then
    echo "[VENV] Creating virtual environment..."
    python3 -m venv --system-site-packages "$PROJECT_ROOT/$VENV_NAME"
else
    echo "[VENV] Virtual environment already exists."
fi

# -----------------------------
# Python deps (NO activation)
# -----------------------------
echo "[PIP] Upgrading pip..."
"$PY" -m pip install --upgrade pip

echo "[PIP] Installing Python requirements..."
"$PY" -m pip install --no-cache-dir -r "$REQS_FILE"

# -----------------------------
# Add alias
# -----------------------------
ALIAS_STRING="alias $ALIAS_NAME='source \"$ACTIVATE_SCRIPT_PATH\"'"
ALIAS_STRING_ESCAPED=$(printf '%s\n' "$ALIAS_STRING" | sed 's/[\/&]/\\&/g')

if ! grep -q "alias $ALIAS_NAME=" "$BASHRC_FILE"; then
    echo "[ALIAS] Adding $ALIAS_NAME to ~/.bashrc"
    echo "" >> "$BASHRC_FILE"
    echo "# AImy virtual environment" >> "$BASHRC_FILE"
    echo "$ALIAS_STRING" >> "$BASHRC_FILE"
else
    echo "[ALIAS] Updating existing $ALIAS_NAME alias"
    sed -i "s|^alias $ALIAS_NAME=.*|$ALIAS_STRING_ESCAPED|" "$BASHRC_FILE"
fi

# -----------------------------
# Download models
# -----------------------------
echo "=============================================="
echo "[MODELS] Downloading Axera model resources..."
echo "=============================================="

"$PY" "$PROJECT_ROOT/scripts/download_models.py"

# -----------------------------
# Done
# -----------------------------
echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Run:"
echo "  source ~/.bashrc"
echo "Then activate anytime with:"
echo "  $ALIAS_NAME"
