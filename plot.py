"""Accuracy against storage and against MACs for the runs in a results file.

    python plot.py results/mobilevit.json --out results/mobilevit.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"prune": "#2a6fdb", "quant": "#e07b39", "pipeline": "#3a9d5d"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results")
    p.add_argument("--out", required=True)
    p.add_argument("--label", action="store_true", help="write the variant name next to each point")
    args = p.parse_args(argv)
    with open(args.results, encoding="utf-8") as f:
        rows = [r for r in json.load(f) if "error" not in r]
    if not rows:
        raise SystemExit("no successful runs")
    base = rows[0]["baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, xlabel, scale in ((axes[0], "size_bytes", "storage (KiB, log)", 1024),
                                   (axes[1], "macs", "MACs (M, log)", 1e6)):
        for kind, color in COLORS.items():
            pts = [r for r in rows if r["kind"] == kind]
            ax.scatter([r["final"][key] / scale for r in pts], [r["final"]["acc"] for r in pts], s=22,
                       color=color, label=kind, alpha=0.85)
            if args.label:
                for r in pts:
                    ax.annotate(r["variant"], (r["final"][key] / scale, r["final"]["acc"]), fontsize=6,
                                xytext=(3, 2), textcoords="offset points")
        ax.scatter([base[key] / scale], [base["acc"]], marker="*", s=160, color="black", label="baseline")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CIFAR-10 top-1 (%)")
        ax.grid(alpha=0.3)
        ax.set_ylim(max(0, min(r["final"]["acc"] for r in rows) - 5), base["acc"] + 3)
    axes[0].legend(loc="lower right")
    fig.suptitle(f"{rows[0]['model']}: accuracy vs storage and compute")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
