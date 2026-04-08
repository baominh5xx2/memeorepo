#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ID="Minhbao5xx2/damp_es"
DEST_ROOT="$PROJECT_ROOT/data"
REVISION="main"
TOKEN="${HF_TOKEN:-}"
SKIP_INSTALL="0"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/fetch_hf_raw_data.sh [options]

Options:
  --repo-id <id>       HF dataset repo id (default: Minhbao5xx2/damp_es)
  --dest-root <path>   Destination folder (default: ./data)
  --revision <rev>     Repo revision/branch/tag (default: main)
  --token <token>      HF token (default: read from HF_TOKEN env)
  --python-bin <bin>   Python executable (default: python3)
  --skip-install       Skip pip install/upgrade huggingface_hub
  -h, --help           Show this help

Behavior:
  1) Try downloading LUAD-HistoSeg.zip and BCSS-WSSS.zip
  2) Extract zip files into --dest-root
  3) If zip files are missing, fallback to folder snapshot download
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    --dest-root)
      DEST_ROOT="$2"
      shift 2
      ;;
    --revision)
      REVISION="$2"
      shift 2
      ;;
    --token)
      TOKEN="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$DEST_ROOT"

if [[ "$SKIP_INSTALL" != "1" ]]; then
  "$PYTHON_BIN" -m pip install --upgrade huggingface_hub
fi

"$PYTHON_BIN" - "$REPO_ID" "$DEST_ROOT" "$REVISION" "$TOKEN" <<'PY'
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

repo_id, dest_root, revision, token = sys.argv[1:5]
token = token or None
dest = Path(dest_root)
dest.mkdir(parents=True, exist_ok=True)

zip_names = ["LUAD-HistoSeg.zip", "BCSS-WSSS.zip"]
downloaded = []

for name in zip_names:
    try:
        p = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=name,
            revision=revision,
            token=token,
            local_dir=str(dest),
        )
        downloaded.append(Path(p))
        print(f"[downloaded] {name} -> {p}")
    except Exception as exc:
        print(f"[warn] Cannot download {name}: {exc}")

if downloaded:
    for zip_path in downloaded:
        print(f"[extract] {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
else:
    print("[warn] Zip files not found. Falling back to folder snapshot download.")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
        local_dir=str(dest),
        allow_patterns=["LUAD-HistoSeg/**", "BCSS-WSSS/**"],
    )

required = [
    dest / "LUAD-HistoSeg" / "LUAD-HistoSeg" / "training",
    dest / "BCSS-WSSS" / "BCSS-WSSS" / "training",
]

missing = [str(p) for p in required if not p.exists()]
if missing:
    print("[error] Required directories are missing:")
    for p in missing:
        print(f"  - {p}")
    raise SystemExit(2)

print(f"[ok] Raw datasets are ready at: {dest}")
PY

echo "Done. Next run:"
echo "  python main.py --source Hist --target BCSS --raw-data-root $DEST_ROOT --dataset-root $DEST_ROOT/CrossDomainSeg --stage2-disable-crf"
