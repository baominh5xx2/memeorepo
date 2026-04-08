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
- `--skip-stage1 --stage1-ckpt <path>` when reusing existing Stage 1 checkpoint.
- `--stage2-disable-crf` if `pydensecrf` is unavailable.
- `--dry-run` to print all stage commands without executing.

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
