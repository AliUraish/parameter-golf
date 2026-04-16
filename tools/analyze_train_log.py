#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


TRAIN_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<iterations>\d+) train_loss:(?P<train_loss>[0-9.]+) "
    r"train_time:(?P<train_time_ms>[0-9.]+)ms step_avg:(?P<step_avg_ms>[0-9.]+)ms"
)
VAL_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<iterations>\d+) val_loss:(?P<val_loss>[0-9.]+) "
    r"val_bpb:(?P<val_bpb>[0-9.]+) train_time:(?P<train_time_ms>[0-9.]+)ms"
)
FINAL_EXACT_RE = re.compile(
    r"^final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+) "
    r"val_bpb:(?P<val_bpb>[0-9.]+)"
)
SERIALIZED_RE = re.compile(r"^serialized_model_int8_zlib:(?P<bytes>[0-9]+) bytes")
SHAPE_RE = re.compile(
    r"^model_params:(?P<model_params>[0-9]+).*layers:(?P<layers>[0-9]+) "
    r"dim:(?P<dim>[0-9]+) heads:(?P<heads>[0-9]+) kv_heads:(?P<kv_heads>[0-9]+)"
)
RUN_ID_RE = re.compile(r"^run_id:(?P<run_id>\S+)")
VAL_SUBSET_RE = re.compile(r"^WARNING: val_loader:subset val_max_seqs:(?P<val_max_seqs>[0-9]+)")
EMA_RE = re.compile(r"^final_weight_ema:(?P<state>enabled|disabled)(?: beta:(?P<beta>[0-9.]+) start_step:(?P<start_step>[0-9]+))?")


@dataclass
class TrainPoint:
    step: int
    iterations: int
    train_loss: float
    train_time_ms: float
    step_avg_ms: float


@dataclass
class ValPoint:
    step: int
    iterations: int
    val_loss: float
    val_bpb: float
    train_time_ms: float


def parse_log(path: Path) -> dict[str, object]:
    info: dict[str, object] = {
        "path": path,
        "run_id": None,
        "shape": None,
        "train_points": [],
        "val_points": [],
        "final_exact_bpb": None,
        "final_exact_loss": None,
        "serialized_bytes": None,
        "val_max_seqs": None,
        "ema": None,
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if info["run_id"] is None:
            match = RUN_ID_RE.search(line)
            if match:
                info["run_id"] = match.group("run_id")

        if info["shape"] is None:
            match = SHAPE_RE.search(line)
            if match:
                info["shape"] = {
                    "model_params": int(match.group("model_params")),
                    "layers": int(match.group("layers")),
                    "dim": int(match.group("dim")),
                    "heads": int(match.group("heads")),
                    "kv_heads": int(match.group("kv_heads")),
                }

        if info["val_max_seqs"] is None:
            match = VAL_SUBSET_RE.search(line)
            if match:
                info["val_max_seqs"] = int(match.group("val_max_seqs"))

        if info["ema"] is None:
            match = EMA_RE.search(line)
            if match:
                info["ema"] = {
                    "enabled": match.group("state") == "enabled",
                    "beta": float(match.group("beta")) if match.group("beta") else None,
                    "start_step": int(match.group("start_step")) if match.group("start_step") else None,
                }

        match = TRAIN_RE.search(line)
        if match:
            info["train_points"].append(
                TrainPoint(
                    step=int(match.group("step")),
                    iterations=int(match.group("iterations")),
                    train_loss=float(match.group("train_loss")),
                    train_time_ms=float(match.group("train_time_ms")),
                    step_avg_ms=float(match.group("step_avg_ms")),
                )
            )
            continue

        match = VAL_RE.search(line)
        if match:
            info["val_points"].append(
                ValPoint(
                    step=int(match.group("step")),
                    iterations=int(match.group("iterations")),
                    val_loss=float(match.group("val_loss")),
                    val_bpb=float(match.group("val_bpb")),
                    train_time_ms=float(match.group("train_time_ms")),
                )
            )
            continue

        match = FINAL_EXACT_RE.search(line)
        if match:
            info["final_exact_loss"] = float(match.group("val_loss"))
            info["final_exact_bpb"] = float(match.group("val_bpb"))
            continue

        match = SERIALIZED_RE.search(line)
        if match:
            info["serialized_bytes"] = int(match.group("bytes"))

    return info


def diagnosis(info: dict[str, object]) -> list[str]:
    notes: list[str] = []
    train_points: list[TrainPoint] = info["train_points"]  # type: ignore[assignment]
    val_points: list[ValPoint] = info["val_points"]  # type: ignore[assignment]
    final_exact_bpb = info["final_exact_bpb"]
    serialized_bytes = info["serialized_bytes"]
    val_max_seqs = info["val_max_seqs"]

    if val_max_seqs:
        notes.append(f"local subset eval only (VAL_MAX_SEQS={val_max_seqs})")
    else:
        notes.append("full validation run")

    if train_points:
        first = train_points[0].train_loss
        last = train_points[-1].train_loss
        if last < first * 0.75:
            notes.append("training is learning")
        if len(train_points) >= 2 and train_points[-1].train_loss < train_points[-2].train_loss:
            notes.append("last logged train loss still improving")

    if val_points and final_exact_bpb is not None:
        best_pre = min(val_points, key=lambda x: x.val_bpb)
        gap = final_exact_bpb - best_pre.val_bpb
        if gap > 0.25:
            notes.append(f"large quantization gap (+{gap:.4f} bpb)")
        elif gap > 0.05:
            notes.append(f"moderate quantization gap (+{gap:.4f} bpb)")
        else:
            notes.append(f"small quantization gap (+{gap:.4f} bpb)")

    if serialized_bytes is not None:
        if serialized_bytes > 16_000_000:
            notes.append("over the 16 MB cap")
        elif serialized_bytes < 12_000_000:
            notes.append("large size headroom remains")
        else:
            notes.append("artifact size is near the challenge budget")

    return notes


def print_summary(info: dict[str, object]) -> None:
    path: Path = info["path"]  # type: ignore[assignment]
    run_id = info["run_id"] or path.stem
    shape = info["shape"]
    train_points: list[TrainPoint] = info["train_points"]  # type: ignore[assignment]
    val_points: list[ValPoint] = info["val_points"]  # type: ignore[assignment]
    final_exact_bpb = info["final_exact_bpb"]
    serialized_bytes = info["serialized_bytes"]
    ema = info["ema"]

    print(f"== {run_id} ==")
    print(path)
    if shape:
        print(
            f"shape: params={shape['model_params']} layers={shape['layers']} dim={shape['dim']} "
            f"heads={shape['heads']} kv={shape['kv_heads']}"
        )
    if ema:
        if ema["enabled"]:
            print(f"ema: enabled beta={ema['beta']} start_step={ema['start_step']}")
        else:
            print("ema: disabled")
    if train_points:
        first = train_points[0]
        last = train_points[-1]
        print(
            f"train: first_loss={first.train_loss:.4f} last_loss={last.train_loss:.4f} "
            f"last_step={last.step}/{last.iterations} step_avg_ms={last.step_avg_ms:.2f}"
        )
    if val_points:
        best_pre = min(val_points, key=lambda x: x.val_bpb)
        last_pre = val_points[-1]
        print(
            f"pre_quant: best_step={best_pre.step} best_val_bpb={best_pre.val_bpb:.4f} "
            f"last_step={last_pre.step} last_val_bpb={last_pre.val_bpb:.4f}"
        )
    if final_exact_bpb is not None:
        print(f"final_exact_val_bpb={final_exact_bpb:.8f}")
    if serialized_bytes is not None:
        print(f"artifact_bytes={serialized_bytes}")
    print("diagnosis: " + "; ".join(diagnosis(info)))
    print()


def expand_paths(raw_paths: list[str]) -> list[Path]:
    if raw_paths:
        return [Path(p).expanduser().resolve() for p in raw_paths]
    logs_dir = Path("logs")
    if not logs_dir.is_dir():
        return []
    return sorted(logs_dir.glob("*.txt"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize parameter-golf training logs")
    parser.add_argument("paths", nargs="*", help="Log files to summarize. Defaults to logs/*.txt")
    args = parser.parse_args()

    paths = expand_paths(args.paths)
    if not paths:
        raise SystemExit("No log files found")

    summaries = [parse_log(path) for path in paths]
    for info in summaries:
        print_summary(info)


if __name__ == "__main__":
    main()
