from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full DAMP-ES pipeline from one Python entrypoint"
    )

    parser.add_argument("--source", type=str, default="Hist", help="Source domain")
    parser.add_argument("--target", type=str, default="BCSS", help="Target domain")
    parser.add_argument(
        "--exp-name",
        type=str,
        default="",
        help="Optional custom experiment name (default: <source>_to_<target>)",
    )

    parser.add_argument("--raw-data-root", type=str, default="data")
    parser.add_argument("--dataset-root", type=str, default="data/CrossDomainSeg")

    parser.add_argument("--stage1-config", type=str, default="configs/stage1_damp.yaml")
    parser.add_argument("--stage2-config", type=str, default="configs/stage2_cam.yaml")
    parser.add_argument("--stage3-config", type=str, default="configs/stage3_seg.yaml")

    parser.add_argument("--stage1-out", type=str, default="")
    parser.add_argument("--stage2-cam-out", type=str, default="")
    parser.add_argument("--stage2-pseudomask-out", type=str, default="")
    parser.add_argument("--stage3-out", type=str, default="")
    parser.add_argument(
        "--stage1-ckpt",
        type=str,
        default="",
        help="Checkpoint root for Stage2 (defaults to --stage1-out)",
    )

    parser.add_argument("--eval-split", type=str, default="test")

    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--stage2-disable-crf", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--stage1-override",
        nargs="*",
        default=[],
        help="Extra key=value overrides appended to Stage1",
    )
    parser.add_argument(
        "--stage2-override",
        nargs="*",
        default=[],
        help="Extra key=value overrides appended to Stage2",
    )
    parser.add_argument(
        "--stage3-override",
        nargs="*",
        default=[],
        help="Extra key=value overrides appended to Stage3",
    )

    return parser.parse_args()


def _build_env() -> dict:
    env = os.environ.copy()
    parent = str(PARENT_ROOT)
    old_pythonpath = env.get("PYTHONPATH", "")

    if not old_pythonpath:
        env["PYTHONPATH"] = parent
    else:
        parts = old_pythonpath.split(os.pathsep)
        if parent not in parts:
            env["PYTHONPATH"] = parent + os.pathsep + old_pythonpath
    return env


def _run_step(step_name: str, script_rel: str, script_args: List[str], dry_run: bool) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / script_rel), *script_args]
    print(f"== {step_name} ==")
    print(" ".join(cmd))
    if dry_run:
        return

    subprocess.run(cmd, cwd=PROJECT_ROOT, env=_build_env(), check=True)


def _exp_name(source: str, target: str, custom_name: str) -> str:
    if custom_name.strip():
        return custom_name.strip()
    return f"{source.lower()}_to_{target.lower()}"


def main() -> None:
    args = parse_args()

    source = args.source
    target = args.target
    exp_name = _exp_name(source=source, target=target, custom_name=args.exp_name)

    stage1_out = args.stage1_out or f"checkpoints/stage1/{exp_name}"
    stage2_cam_out = args.stage2_cam_out or f"outputs/stage2/cams/{exp_name}"
    stage2_pseudomask_out = args.stage2_pseudomask_out or f"outputs/stage2/pseudomasks/{exp_name}"
    stage3_out = args.stage3_out or f"checkpoints/stage3/{exp_name}"
    stage1_ckpt = args.stage1_ckpt or stage1_out

    if not args.skip_prepare:
        _run_step(
            step_name="Phase 1: Prepare datasets",
            script_rel="tools/prepare_medical_datasets.py",
            script_args=[
                "--raw-data-root",
                args.raw_data_root,
                "--output-root",
                args.dataset_root,
            ],
            dry_run=args.dry_run,
        )
    else:
        print("== Phase 1: Prepare datasets skipped ==")

    if not args.skip_stage1:
        stage1_override = [
            f"dataset.source_domain={source}",
            f"dataset.target_domain={target}",
            f"dataset.root={args.dataset_root}",
            f"training.output_dir={stage1_out}",
            *args.stage1_override,
        ]
        _run_step(
            step_name="Stage 1: DAMP adaptation",
            script_rel="stage1_damp/train.py",
            script_args=["--config", args.stage1_config, "--override", *stage1_override],
            dry_run=args.dry_run,
        )
    else:
        print("== Stage 1 skipped ==")

    stage2_override = [
        f"dataset.target_domain={target}",
        f"dataset.root={args.dataset_root}",
        f"output.cam_dir={stage2_cam_out}",
        f"output.pseudomask_dir={stage2_pseudomask_out}",
        f"model.damp_ckpt={stage1_ckpt}",
        *args.stage2_override,
    ]
    if args.stage2_disable_crf:
        stage2_override.append("crf.enabled=false")

    _run_step(
        step_name="Stage 2: CAM + pseudo-mask",
        script_rel="stage2_cam/generate_pseudomasks.py",
        script_args=["--config", args.stage2_config, "--override", *stage2_override],
        dry_run=args.dry_run,
    )

    stage3_override = [
        f"dataset.root={args.dataset_root}",
        f"dataset.domain={target}",
        f"dataset.pseudo_mask_dir={stage2_pseudomask_out}",
        f"training.save_dir={stage3_out}",
        *args.stage3_override,
    ]
    _run_step(
        step_name="Stage 3: Segmentation retraining",
        script_rel="stage3_seg/train_seg.py",
        script_args=["--config", args.stage3_config, "--override", *stage3_override],
        dry_run=args.dry_run,
    )

    if not args.skip_eval:
        target_domain_root = f"{args.dataset_root}/{target}"
        _run_step(
            step_name="Final evaluation",
            script_rel="tools/eval_crossdomain.py",
            script_args=[
                "--pred-dir",
                f"{stage3_out}/test_predictions",
                "--domain-root",
                target_domain_root,
                "--split",
                args.eval_split,
            ],
            dry_run=args.dry_run,
        )
    else:
        print("== Final evaluation skipped ==")

    print(f"Pipeline completed for {source} -> {target}")


if __name__ == "__main__":
    main()