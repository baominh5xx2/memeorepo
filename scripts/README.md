# DAMP-ES Run Guide

This repository now keeps a single full-dataset runner: `damp_es/main.py`.

Stage 1 expects the official DAMP repo to be available at one of these locations:
- `./DAMP` (inside `damp_es`)
- `../DAMP` (adjacent to `damp_es`)

## Environment setup

Linux server setup:

```bash
chmod +x scripts/setup_env.sh
# Optional for NVIDIA GPU servers (example: CUDA 12.1 wheels)
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
# Optional: try installing CRF dependency (pydensecrf)
# export INSTALL_CRF=1
./scripts/setup_env.sh
```

`scripts/setup_env.sh` installs dependencies from `requirements.txt`.
CRF dependency is optional and moved to `requirements-crf.txt`.

If you want CRF and build fails, install compiler toolchain first (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y build-essential g++ python3-dev
```

## Single full-dataset command

From the damp_es folder (required):

```bash
pwd
# should end with /damp_es
mkdir -p logs
```

```bash
python main.py --source Hist --target BCSS --exp-name real_hist_to_bcss
```

```powershell
python main.py --source Hist --target BCSS --exp-name real_hist_to_bcss
```

Useful options:
- `--skip-prepare` when `data/CrossDomainSeg` is already prepared.
- `--prepare-workers 64 --prepare-domain-workers 2` to speed up Phase 1 preprocessing.
- `--skip-stage1 --stage1-ckpt <path>` when reusing existing Stage 1 checkpoint.
- `--stage2-disable-crf` if `pydensecrf` is unavailable.
- `--dry-run` to print all stage commands without executing.

Fast prepare example (parallel copy/remap):

```bash
python main.py --source Hist --target BCSS --raw-data-root ./data --dataset-root ./data/CrossDomainSeg --prepare-workers 64 --prepare-domain-workers 2 --stage2-disable-crf
```

## Troubleshooting

If you see this error:

```text
ModuleNotFoundError: No module named 'damp_es'
```

and traceback paths look like `/home/admin/main.py` (instead of `/home/admin/damp_es/main.py`),
your deployment is likely flattened at server root.

Quick fix without re-uploading:

```bash
cd /home/admin
mkdir -p logs
ln -sfn . damp_es
export PYTHONPATH=/home/admin:${PYTHONPATH:-}
python main.py --source Hist --target BCSS --exp-name real_hist_to_bcss --stage2-disable-crf 2>&1 | tee logs/real_hist_to_bcss.log
```

Recommended clean layout:

```text
/home/admin/damp_es/main.py
/home/admin/damp_es/tools/
/home/admin/damp_es/stage1_damp/
...
```

## Download raw datasets from Hugging Face

If you uploaded `LUAD-HistoSeg.zip` and `BCSS-WSSS.zip` to a dataset repo,
use this script on server to fetch and extract them into `data/`:

```bash
chmod +x scripts/fetch_hf_raw_data.sh
export HF_TOKEN=hf_xxx
./scripts/fetch_hf_raw_data.sh --repo-id Minhbao5xx2/damp_es --dest-root ./data
```

Then run pipeline:

```bash
python main.py --source Hist --target BCSS --raw-data-root ./data --dataset-root ./data/CrossDomainSeg --stage2-disable-crf
```
