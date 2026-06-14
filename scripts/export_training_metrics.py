"""Export Trainer log history to CSV and SVG plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape


METRIC_COLUMNS = [
    "step",
    "epoch",
    "loss",
    "mean_token_accuracy",
    "grad_norm",
    "learning_rate",
    "entropy",
    "num_tokens",
]


def main() -> None:
    """Exports log history from a Trainer checkpoint or run directory."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Training run directory containing checkpoint-* directories.",
    )
    parser.add_argument(
        "--state",
        help="Explicit trainer_state.json path. Defaults to latest checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for metrics CSV and SVG plots.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state) if args.state else _latest_state_path(run_dir)
    history = _load_history(state_path)

    csv_path = output_dir / "training_metrics.csv"
    _write_csv(history, csv_path)
    _write_svg(
        history,
        output_dir / "loss.svg",
        y_key="loss",
        title="Training Loss",
        y_label="loss",
    )
    _write_svg(
        history,
        output_dir / "mean_token_accuracy.svg",
        y_key="mean_token_accuracy",
        title="Mean Token Accuracy",
        y_label="accuracy",
    )
    _write_svg(
        history,
        output_dir / "learning_rate.svg",
        y_key="learning_rate",
        title="Learning Rate",
        y_label="learning rate",
    )
    summary = {
        "state_path": str(state_path),
        "records": len(history),
        "csv_path": str(csv_path),
        "plots": [
            str(output_dir / "loss.svg"),
            str(output_dir / "mean_token_accuracy.svg"),
            str(output_dir / "learning_rate.svg"),
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _latest_state_path(run_dir: Path) -> Path:
    checkpoints = [
        path
        for path in run_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"no numeric checkpoint directories under {run_dir}")
    checkpoint = max(checkpoints, key=lambda path: int(path.name.removeprefix("checkpoint-")))
    state_path = checkpoint / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"missing trainer state: {state_path}")
    return state_path


def _load_history(state_path: Path) -> list[dict[str, float]]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    records = []
    for record in state.get("log_history", []):
        if "step" not in record:
            continue
        filtered = {}
        for column in METRIC_COLUMNS:
            value = record.get(column)
            if isinstance(value, int | float):
                filtered[column] = value
        if "loss" in filtered or "mean_token_accuracy" in filtered:
            records.append(filtered)
    if not records:
        raise ValueError(f"no scalar log history records found in {state_path}")
    return records


def _write_csv(history: list[dict[str, float]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in history:
            writer.writerow(record)


def _write_svg(
    history: list[dict[str, float]],
    path: Path,
    *,
    y_key: str,
    title: str,
    y_label: str,
) -> None:
    points = [
        (float(record["step"]), float(record[y_key]))
        for record in history
        if "step" in record and y_key in record
    ]
    if not points:
        return

    width = 920
    height = 420
    left = 76
    right = 24
    top = 44
    bottom = 58
    min_x, max_x = _range(point[0] for point in points)
    min_y, max_y = _range(point[1] for point in points)
    if min_y == max_y:
        min_y -= 1.0
        max_y += 1.0
    pad_y = (max_y - min_y) * 0.08
    min_y -= pad_y
    max_y += pad_y

    def scale_x(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * (width - left - right)

    def scale_y(value: float) -> float:
        return top + (max_y - value) / (max_y - min_y) * (height - top - bottom)

    polyline = " ".join(f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in points)
    y_ticks = _ticks(min_y, max_y, count=5)
    x_ticks = _ticks(min_x, max_x, count=6)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.0f}" y="24" text-anchor="middle" font-family="Arial" font-size="18" fill="#111827">{escape(title)}</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#374151"/>',
    ]
    for tick in y_ticks:
        y = scale_y(tick)
        elements.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{tick:.4g}</text>',
            ]
        )
    for tick in x_ticks:
        x = scale_x(tick)
        elements.extend(
            [
                f'<line x1="{x:.2f}" y1="{height - bottom}" x2="{x:.2f}" y2="{height - bottom + 5}" stroke="#374151"/>',
                f'<text x="{x:.2f}" y="{height - bottom + 22}" text-anchor="middle" font-family="Arial" font-size="12" fill="#4b5563">{tick:.0f}</text>',
            ]
        )
    elements.extend(
        [
            f'<polyline fill="none" stroke="#2563eb" stroke-width="2.2" points="{polyline}"/>',
            f'<text x="{width / 2:.0f}" y="{height - 14}" text-anchor="middle" font-family="Arial" font-size="13" fill="#374151">step</text>',
            f'<text transform="translate(18 {height / 2:.0f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13" fill="#374151">{escape(y_label)}</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _range(values) -> tuple[float, float]:
    collected = list(values)
    minimum = min(collected)
    maximum = max(collected)
    if minimum == maximum:
        return minimum - 1.0, maximum + 1.0
    return minimum, maximum


def _ticks(minimum: float, maximum: float, *, count: int) -> list[float]:
    if count <= 1:
        return [minimum]
    step = (maximum - minimum) / (count - 1)
    return [minimum + index * step for index in range(count)]


if __name__ == "__main__":
    main()
