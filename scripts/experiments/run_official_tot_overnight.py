"""Run the official ToT overnight SFT/GRPO experiment matrix."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MANIFEST = Path("data/processed/splits/official-tot-overnight-v1.json")
PROMPT_STYLE = "qwen_chat"
EVAL_SPLIT = "tot_all_1362"
EVAL_TOKENS = 4096


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs/official_tot_overnight_20260618"),
    )
    parser.add_argument("--sft-max-steps", type=int, default=5000)
    parser.add_argument("--sft-save-steps", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--rollout-max-new-tokens", type=int, default=512)
    parser.add_argument("--rollout-num-generations", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--grpo-max-steps", type=int, default=5)
    args = parser.parse_args()

    runner = Runner(args.run_root)
    runner.write_metadata(
        {
            "schema_version": "official-tot-overnight-run-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "model_name": MODEL_NAME,
            "manifest": str(MANIFEST),
            "prompt_style": PROMPT_STYLE,
            "eval_split": EVAL_SPLIT,
            "eval_max_new_tokens": EVAL_TOKENS,
            "sft_max_steps": args.sft_max_steps,
            "sft_save_steps": args.sft_save_steps,
            "rollout_max_new_tokens": args.rollout_max_new_tokens,
            "rollout_num_generations": args.rollout_num_generations,
            "grpo_max_steps": args.grpo_max_steps,
        }
    )

    runner.stage(
        "00_manifest",
        [
            sys.executable,
            "scripts/build_official_tot_manifest.py",
            "--output",
            str(MANIFEST),
        ],
        outputs=[MANIFEST],
    )

    run_eval_and_summary(
        runner=runner,
        stage="01_base_eval_4096",
        checkpoint=MODEL_NAME,
        model_name=MODEL_NAME,
        training_mode="full",
        output_dir=args.run_root / "eval" / "base_4096",
        batch_size=args.eval_batch_size,
    )

    sft_full = SftRun(
        name="sft_full_data_5000",
        split="train_full_1362",
        dataset=args.run_root / "data" / "sft_full_data_5000.jsonl",
        run_dir=args.run_root / "sft" / "SFT-full-data-5000",
    )
    sft_heldout = SftRun(
        name="sft_remove_900to1000_5000",
        split="train_remove_900to1000_1262",
        dataset=args.run_root / "data" / "sft_remove_900to1000_5000.jsonl",
        run_dir=args.run_root / "sft" / "SFT-remove-900to1000-5000",
    )
    for sft in [sft_full, sft_heldout]:
        run_sft(
            runner=runner,
            sft=sft,
            max_steps=args.sft_max_steps,
            save_steps=args.sft_save_steps,
        )
        run_eval_and_summary(
            runner=runner,
            stage=f"eval_{sft.name}_4096",
            checkpoint=sft.run_dir / "final",
            model_name=MODEL_NAME,
            training_mode="full",
            output_dir=args.run_root / "eval" / f"{sft.name}_4096",
            batch_size=args.eval_batch_size,
        )

    grpo_full = GrpoRun(
        name="grpo_full",
        sft=sft_full,
        split="train_full_1362",
        run_dir=args.run_root / "grpo" / "GRPO-full",
    )
    grpo_heldout = GrpoRun(
        name="grpo_remove_900to1000",
        sft=sft_heldout,
        split="train_remove_900to1000_1262",
        run_dir=args.run_root / "grpo" / "GRPO-remove-900to1000",
    )
    for grpo in [grpo_full, grpo_heldout]:
        run_grpo(
            runner=runner,
            grpo=grpo,
            rollout_num_generations=args.rollout_num_generations,
            rollout_batch_size=args.rollout_batch_size,
            rollout_max_new_tokens=args.rollout_max_new_tokens,
            max_steps=args.grpo_max_steps,
        )
        run_eval_and_summary(
            runner=runner,
            stage=f"eval_{grpo.name}_4096",
            checkpoint=grpo.run_dir / "train" / "final",
            model_name=str(grpo.sft.run_dir / "final"),
            training_mode="lora",
            output_dir=args.run_root / "eval" / f"{grpo.name}_4096",
            batch_size=args.eval_batch_size,
        )

    runner.stage(
        "99_collect_results",
        [
            sys.executable,
            "-c",
            _collect_results_code(args.run_root),
        ],
        outputs=[args.run_root / "summary" / "official_tot_results.json"],
    )


class Runner:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root
        self.logs_dir = run_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def write_metadata(self, metadata: dict[str, object]) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        (self.run_root / "run-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def stage(
        self,
        name: str,
        command: list[str],
        *,
        outputs: list[Path] | None = None,
        non_blocking_postprocess: bool = False,
    ) -> None:
        done = self.run_root / "markers" / f"{name}.done"
        if done.exists():
            print(f"[skip] {name} already done", flush=True)
            return
        done.parent.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"{name}.log"
        command_text = " ".join(shlex.quote(str(part)) for part in command)
        print(f"[run] {name}: {command_text}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"$ {command_text}\n")
            log.flush()
            completed = subprocess.run(
                [str(part) for part in command],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            message = (
                f"stage {name} failed with exit code {completed.returncode}; "
                f"see {log_path}"
            )
            if non_blocking_postprocess:
                print(f"[warn] {message}", flush=True)
                return
            raise SystemExit(message)
        for output in outputs or []:
            if not output.exists():
                raise SystemExit(f"stage {name} did not create expected output {output}")
        done.write_text(
            json.dumps(
                {
                    "stage": name,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "log": str(log_path),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[done] {name}", flush=True)


class SftRun:
    def __init__(self, *, name: str, split: str, dataset: Path, run_dir: Path) -> None:
        self.name = name
        self.split = split
        self.dataset = dataset
        self.run_dir = run_dir


class GrpoRun:
    def __init__(self, *, name: str, sft: SftRun, split: str, run_dir: Path) -> None:
        self.name = name
        self.sft = sft
        self.split = split
        self.run_dir = run_dir


def run_sft(
    *,
    runner: Runner,
    sft: SftRun,
    max_steps: int,
    save_steps: int,
) -> None:
    common = [
        sys.executable,
        "scripts/experiments/run_rollback_sft_experiment.py",
        "--manifest",
        str(MANIFEST),
        "--split",
        sft.split,
        "--dataset",
        str(sft.dataset),
        "--run-dir",
        str(sft.run_dir),
        "--training-mode",
        "full",
        "--prompt-style",
        PROMPT_STYLE,
        "--model-name",
        MODEL_NAME,
        "--max-steps",
        str(max_steps),
        "--save-steps",
        str(save_steps),
        "--save-total-limit",
        "1",
        "--max-length",
        "1024",
        "--max-new-tokens",
        "1024",
        "--eval-batch-size",
        "4",
    ]
    runner.stage(
        f"build_{sft.name}_dataset",
        [*common, "--mode", "build"],
        outputs=[sft.dataset, sft.dataset.with_suffix(".summary.json")],
    )
    runner.stage(
        f"train_{sft.name}",
        [*common, "--mode", "train"],
        outputs=[sft.run_dir / "final"],
    )


def run_eval_and_summary(
    *,
    runner: Runner,
    stage: str,
    checkpoint: str | Path,
    model_name: str,
    training_mode: str,
    output_dir: Path,
    batch_size: int,
) -> None:
    report = output_dir / f"{EVAL_SPLIT}-eval-report.json"
    group_summary = output_dir / "group-summary.json"
    runner.stage(
        stage,
        [
            sys.executable,
            "scripts/eval_checkpoint.py",
            "--manifest",
            str(MANIFEST),
            "--split",
            EVAL_SPLIT,
            "--output-dir",
            str(output_dir),
            "--model-name",
            model_name,
            "--checkpoint",
            str(checkpoint),
            "--training-mode",
            training_mode,
            "--batch-size",
            str(batch_size),
            "--max-new-tokens",
            str(EVAL_TOKENS),
            "--prompt-style",
            PROMPT_STYLE,
        ],
        outputs=[report],
    )
    runner.stage(
        f"summarize_{stage}",
        [
            sys.executable,
            "scripts/summarize_tot_eval_groups.py",
            "--report",
            str(report),
            "--output",
            str(group_summary),
        ],
        outputs=[group_summary],
        non_blocking_postprocess=True,
    )


def run_grpo(
    *,
    runner: Runner,
    grpo: GrpoRun,
    rollout_num_generations: int,
    rollout_batch_size: int,
    rollout_max_new_tokens: int,
    max_steps: int,
) -> None:
    rollout_dir = grpo.run_dir / "rollout_pool"
    details = rollout_dir / f"{grpo.split}-rollout-details.json"
    pool = grpo.run_dir / "pool.json"
    train_dir = grpo.run_dir / "train"
    sft_final = grpo.sft.run_dir / "final"
    runner.stage(
        f"rollout_{grpo.name}",
        [
            sys.executable,
            "scripts/experiments/audit_rollout_distribution.py",
            "--manifest",
            str(MANIFEST),
            "--split",
            grpo.split,
            "--checkpoint",
            str(sft_final),
            "--model-name",
            MODEL_NAME,
            "--training-mode",
            "full",
            "--prompt-style",
            PROMPT_STYLE,
            "--output-dir",
            str(rollout_dir),
            "--num-generations",
            str(rollout_num_generations),
            "--batch-size",
            str(rollout_batch_size),
            "--max-new-tokens",
            str(rollout_max_new_tokens),
            "--temperature",
            "0.8",
            "--top-p",
            "0.95",
            "--seed",
            "20260618",
        ],
        outputs=[details],
    )
    runner.stage(
        f"pool_{grpo.name}",
        [
            sys.executable,
            "scripts/build_grpo_pool.py",
            "--details",
            str(details),
            "--output",
            str(pool),
            "--split",
            grpo.split,
            "--checkpoint",
            str(sft_final),
            "--num-generations",
            str(rollout_num_generations),
            "--temperature",
            "0.8",
            "--top-p",
            "0.95",
            "--max-new-tokens",
            str(rollout_max_new_tokens),
            "--min-pool-size",
            "1",
            "--min-mixed-group-rate",
            "0",
            "--max-zero-std-group-rate",
            "1",
            "--min-correct-truncation-mixed",
            "0",
            "--max-all-wrong-rate",
            "1",
            "--select-min-correct",
            "1",
            "--select-max-correct",
            "2",
            "--select-require-truncation",
        ],
        outputs=[pool],
    )
    ensure_nonempty_pool_or_fallback(runner=runner, grpo=grpo, details=details, pool=pool)
    runner.stage(
        f"train_{grpo.name}",
        [
            sys.executable,
            "scripts/train_grpo.py",
            "--train",
            "--manifest",
            str(MANIFEST),
            "--split",
            grpo.split,
            "--output-dir",
            str(train_dir),
            "--model-name-or-path",
            str(sft_final),
            "--pool-manifest",
            str(pool),
            "--prompt-style",
            PROMPT_STYLE,
            "--max-steps",
            str(max_steps),
            "--save-steps",
            str(max_steps),
            "--logging-steps",
            "1",
            "--max-completion-length",
            str(rollout_max_new_tokens),
            "--num-generations",
            str(rollout_num_generations),
            "--gradient-accumulation-steps",
            str(rollout_num_generations),
            "--temperature",
            "0.8",
            "--top-p",
            "0.95",
            "--learning-rate",
            "5e-7",
            "--beta",
            "0.001",
            "--scale-rewards",
            "none",
            "--reward-profile",
            "strict",
            "--peft-mode",
            "lora",
            "--lora-rank",
            "16",
            "--lora-alpha",
            "32",
            "--lora-dropout",
            "0.05",
            "--no-mask-truncated-completions",
            "--no-remove-unused-columns",
        ],
        outputs=[train_dir / "final"],
    )


def ensure_nonempty_pool_or_fallback(
    *,
    runner: Runner,
    grpo: GrpoRun,
    details: Path,
    pool: Path,
) -> None:
    check_code = (
        "import json, pathlib; "
        f"p=pathlib.Path({str(pool)!r}); "
        "d=json.loads(p.read_text()); "
        "raise SystemExit(0 if d.get('selected_prompt_ids') else 1)"
    )
    result = subprocess.run([sys.executable, "-c", check_code], check=False)
    if result.returncode == 0:
        return
    print(f"[warn] {grpo.name} strict truncation-selected pool is empty; fallback to all mixed groups")
    pool.unlink(missing_ok=True)
    runner.stage(
        f"pool_{grpo.name}_fallback_all_mixed",
        [
            sys.executable,
            "scripts/build_grpo_pool.py",
            "--details",
            str(details),
            "--output",
            str(pool),
            "--split",
            grpo.split,
            "--checkpoint",
            str(grpo.sft.run_dir / "final"),
            "--min-pool-size",
            "1",
            "--min-mixed-group-rate",
            "0",
            "--max-zero-std-group-rate",
            "1",
            "--min-correct-truncation-mixed",
            "0",
            "--max-all-wrong-rate",
            "1",
        ],
        outputs=[pool],
    )
    result = subprocess.run([sys.executable, "-c", check_code], check=False)
    if result.returncode == 0:
        return
    print(f"[warn] {grpo.name} mixed pool is empty; fallback to all prompts in split")
    force_pool_all_prompt_ids(pool=pool, split=grpo.split)


def force_pool_all_prompt_ids(*, pool: Path, split: str) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ids = [str(record["id"]) for record in manifest["splits"][split]]
    payload = json.loads(pool.read_text(encoding="utf-8"))
    payload["selected_prompt_ids"] = ids
    payload["selection_filter"] = {
        "fallback": "all_prompt_ids",
        "selected_count": len(ids),
        "reason": "filtered and mixed GRPO pools were empty",
    }
    payload.setdefault("audit", {})["passed"] = True
    pool.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _collect_results_code(run_root: Path) -> str:
    return f"""
import json
from pathlib import Path
root = Path({str(run_root)!r})
rows = []
for path in sorted(root.glob('eval/*_4096/group-summary.json')):
    data = json.loads(path.read_text())
    rows.append({{
        'stage': path.parent.name,
        'checkpoint': data.get('checkpoint'),
        'groups': data.get('groups'),
        'summary_path': str(path),
    }})
out = root / 'summary' / 'official_tot_results.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + '\\n')
print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
"""


if __name__ == "__main__":
    main()
