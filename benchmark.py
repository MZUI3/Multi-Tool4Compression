"""Run many configs on one or more baselines and write a results table.

    python benchmark.py --suite prune --models mobilevit
    python benchmark.py --configs magnitude gmp l1_filter ptq_minmax --models mobilevit resnet18
    python benchmark.py --suite all --models mobilevit --name mobilevit_all

``--suite`` is a folder of configs/ (prune, quant, pipeline) or ``all``. A config may list
``applies_to: [model, ...]``; other models skip it. Results go to results/<name>.json and .md;
``--resume`` skips (variant, model, seed) entries already in that JSON.
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from mtcompress.config import CONFIG_DIR, build_config, load_yaml  # noqa: E402


def config_names(suite):
    folders = ["prune", "quant", "pipeline"] if suite == "all" else [suite]
    out = []
    for f in folders:
        out += sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(CONFIG_DIR, f, "*.yaml")))
    return out


def fmt_kib(b):
    return f"{b / 1024:.1f}"


def table(rows):
    head = ("| Variant | Method | Acc (Δ) | Weight sparsity | Size KiB (×) | MACs M | Sparse MACs M | "
            "PyTorch CPU ms | ORT fp32 ms | ORT int8 ms | Time s |")
    sep = "|" + "---|" * 11
    lines = [head, sep]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['variant']} | {r.get('method', '')} | error: {r['error'][:60]} | | | | | | | | |")
            continue
        f = dict(r["final"])
        f["compression_ratio"] = r["baseline"]["size_bytes"] / max(f["size_bytes"], 1.0)

        def ms(v):
            return f"{v:.3f}" if v else "n/a"

        lines.append(
            f"| {r['variant']} | {r['method']} | {f['acc']:.2f} ({r['delta_acc']:+.2f}) | {f['weight_sparsity']:.1f}% | "
            f"{fmt_kib(f['size_bytes'])} (×{f['compression_ratio']:.2f}) | {f['macs'] / 1e6:.2f} | "
            f"{f['sparse_macs'] / 1e6:.2f} | {ms(f.get('torch_cpu_latency_ms'))} | "
            f"{ms(f.get('onnx_fp32_latency_ms'))} | {ms(f.get('onnx_int8_latency_ms'))} | {r['wall_time_s']:.0f} |")
    return "\n".join(lines)


def write_markdown(path, rows):
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    parts = []
    for model, rs in by_model.items():
        ok = [r for r in rs if "error" not in r]
        if ok:
            b = ok[0]["baseline"]
            tl = b.get("torch_cpu_latency_ms")
            ol = b.get("onnx_fp32_latency_ms")
            il = b.get("onnx_int8_latency_ms")
            lat = (f", PyTorch CPU {tl:.3f} ms, ONNX Runtime CPU {ol:.3f} ms fp32 / {il:.3f} ms int8 "
                   f"(batch 1, 4 threads)" if tl and ol and il else "")
            parts.append(f"## {model}\n\nBaseline: {b['acc']:.2f}% top-1, {b['params']:,} parameters, "
                         f"{fmt_kib(b['size_bytes'])} KiB fp32, {b['macs'] / 1e6:.2f}M MACs{lat}.\n")
        parts.append(table(rs) + "\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--suite", choices=["prune", "quant", "pipeline", "all"])
    g.add_argument("--configs", nargs="+")
    g.add_argument("--render", metavar="NAME", help="only rewrite results/NAME.md from results/NAME.json")
    p.add_argument("--models", nargs="+", default=["mobilevit"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--name", help="results/<name>.json|md (default: suite or 'custom')")
    p.add_argument("--save-dir", default=None, help="run dirs (default runs/benchmark/<timestamp>)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)
    if args.render:
        with open(os.path.join(ROOT, "results", f"{args.render}.json"), encoding="utf-8") as f:
            write_markdown(os.path.join(ROOT, "results", f"{args.render}.md"), json.load(f))
        print(f"wrote results/{args.render}.md")
        return

    from mtcompress.runner import run

    names = config_names(args.suite) if args.suite else args.configs
    out_name = args.name or (args.suite or "custom")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json_path = os.path.join(ROOT, "results", f"{out_name}.json")
    rows = []
    if args.resume and os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as f:
            rows = [r for r in json.load(f) if "error" not in r]
    done = {(r["variant"], r["model"], r["seed"]) for r in rows}
    save_dir = args.save_dir or os.path.join("runs", "benchmark", time.strftime("%Y%m%d-%H%M%S"))
    for model in args.models:
        for name in names:
            for seed in args.seeds:
                cfg = build_config(None, name, {"model": model, "seed": seed, "save_dir": save_dir}, args.set)
                applies = cfg.pop("applies_to", None)
                if applies and model not in applies:
                    continue
                if (cfg["variant"], model, seed) in done:
                    continue
                print(f"\n===== {cfg['variant']} / {model} / seed {seed}", flush=True)
                try:
                    s = run(cfg, log=lambda *a, **k: print(*a, **k, flush=True))
                    s.pop("info", None)
                    rows.append(s)
                except Exception as exc:
                    traceback.print_exc()
                    rows.append({"variant": cfg["variant"], "method": cfg["method"], "model": model, "seed": seed,
                                 "error": f"{type(exc).__name__}: {exc}"})
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(rows, f, indent=1, default=str)
                write_markdown(os.path.join(ROOT, "results", f"{out_name}.md"), rows)
    print(f"\nwrote results/{out_name}.json and .md")


if __name__ == "__main__":
    main()
