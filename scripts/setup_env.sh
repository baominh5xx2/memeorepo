#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set TORCH_INDEX_URL for GPU wheel channel when needed.
# Example (CUDA 12.1):
#   export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

# Optional CRF installation. Set INSTALL_CRF=1 to attempt installing pydensecrf.
# If build fails, setup continues unless CRF_STRICT=1.
INSTALL_CRF="${INSTALL_CRF:-0}"
CRF_STRICT="${CRF_STRICT:-0}"

echo "Using python: $PYTHON_BIN"
"$PYTHON_BIN" -m pip install --upgrade pip

if [[ -n "$TORCH_INDEX_URL" ]]; then
  "$PYTHON_BIN" -m pip install --extra-index-url "$TORCH_INDEX_URL" torch torchvision
fi

"$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements.txt"

if [[ "$INSTALL_CRF" == "1" ]]; then
  echo "Attempting optional CRF dependencies install (pydensecrf)..."
  if "$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements-crf.txt"; then
    echo "Optional CRF dependencies installed successfully."
  else
    echo "WARNING: Optional CRF install failed (likely missing g++/python3-dev)."
    echo "         You can still run pipeline with --stage2-disable-crf."
    if [[ "$CRF_STRICT" == "1" ]]; then
      echo "CRF_STRICT=1, exiting due to CRF install failure."
      exit 1
    fi
  fi
fi

echo "Environment setup completed."
