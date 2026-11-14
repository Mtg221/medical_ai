"""Run learning-curve experiment (100/200/400/600 docs) on SYNTH-MED-V3.1 val split."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.eval_harness.runner import run_experiment

SIZES: tuple[int, ...] = (100, 200, 400, 600)
APPROACHES: tuple[str, ...] = ("classical_crf", "bilstm_crf", "transformer", "hybrid")
COLORS: dict[str, str] = {
    "classical_crf": "#2563eb",
    "bilstm_crf": "#16a34a",
    "transformer": "#d97706",
    "hybrid": "#dc2626",
}


def _render_svg(results: dict[str, Any], out_svg: Path) -> None:
    w, h = 760, 420
    pad_l, pad_r, pad_t, pad_b = 70, 170, 40, 55
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    def x_pos(n: int) -> float:
        return pad_l + ((n - 100) / 500.0) * plot_w

    def y_pos(f1: float) -> float:
        return pad_t + (1.0 - max(0.0, min(1.0, f1))) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{pad_l + plot_w / 2:.1f}" y="24" text-anchor="middle" '
        'font-family="sans-serif" font-size="15" font-weight="bold" fill="#111827">'
        "SYNTH-MED-V3.1 Learning Curve (val, 160 docs)</text>",
    ]
    # Grid & axes
    for tick in (0.2, 0.4, 0.6, 0.8, 1.0):
        yy = y_pos(tick)
        lines.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l + plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>'
        )
        lines.append(
            f'<text x="{pad_l - 8}" y="{yy + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11" fill="#4b5563">{tick:.1f}</text>'
        )
    for n in SIZES:
        xx = x_pos(n)
        lines.append(
            f'<line x1="{xx:.1f}" y1="{pad_t}" x2="{xx:.1f}" y2="{pad_t + plot_h}" stroke="#f3f4f6"/>'
        )
        lines.append(
            f'<text x="{xx:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11" fill="#4b5563">{n}</text>'
        )
    lines.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        'fill="none" stroke="#9ca3af"/>'
    )
    lines.append(
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{h - 12}" text-anchor="middle" '
        'font-family="sans-serif" font-size="12" fill="#374151">Training Documents</text>'
    )

    for idx, app in enumerate(APPROACHES):
        pts = []
        color = COLORS[app]
        for row in results["curves"][app]:
            xx = x_pos(row["train_docs"])
            yy = y_pos(row["strict_f1"])
            pts.append(f"{xx:.1f},{yy:.1f}")
            lines.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{color}"/>')
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(pts)}"/>'
        )
        ly = pad_t + 20 + idx * 24
        lx = pad_l + plot_w + 18
        lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 20}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>')
        lines.append(
            f'<text x="{lx + 26}" y="{ly + 4}" font-family="sans-serif" font-size="12" fill="#111827">{app}</text>'
        )

    lines.append("</svg>")
    out_svg.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results: dict[str, Any] = {
        "dataset": "SYNTH-MED-V3.1",
        "split": "val",
        "num_eval_docs": 160,
        "train_sizes": list(SIZES),
        "seed": 1,
        "curves": {app: [] for app in APPROACHES},
    }
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        for app in APPROACHES:
            base_cfg = yaml.safe_load(Path(f"configs/exp/v3_1/{app}.yaml").read_text(encoding="utf-8"))
            for n_docs in SIZES:
                cfg = dict(base_cfg)
                cfg["max_train_docs"] = n_docs
                cfg["eval_split"] = "val"
                cfg["seeds"] = [1]
                tmp_cfg = tmp_dir / f"{app}_{n_docs}.yaml"
                tmp_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")
                res = run_experiment(tmp_cfg, seeds=[1])
                s = res["summary"]
                ps0 = res["per_seed"][0]
                peak_ram = float(ps0.get("resources", {}).get("peak_rss_mb", 0.0))
                row = {
                    "train_docs": n_docs,
                    "strict_f1": round(float(s["strict_f1"]["mean"]), 4),
                    "lenient_f1": round(float(s["lenient_f1"]["mean"]), 4),
                    "doc_type_acc": round(float(ps0["metrics"]["doc_classification"]["accuracy"]), 4),
                    "wall_seconds": round(float(s["wall_seconds"]["mean"]), 2),
                    "peak_rss_mb": round(peak_ram, 2),
                }
                results["curves"][app].append(row)
                print(
                    f"[learning_curve] {app:<14} n={n_docs:<3} | Strict={row['strict_f1']:.4f} "
                    f"| Lenient={row['lenient_f1']:.4f} | DocAcc={row['doc_type_acc']:.4f} "
                    f"| wall={row['wall_seconds']:.1f}s | RAM={row['peak_rss_mb']:.1f}MB",
                    flush=True,
                )

    out_json = Path("docs/results/v3_1_learning_curve.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    out_svg = Path("docs/results/v3_1_learning_curve.svg")
    _render_svg(results, out_svg)
    print(f"Wrote {out_json} and {out_svg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
