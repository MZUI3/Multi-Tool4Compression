"""Compress a trained baseline with one method (or a pipeline) and measure the result.

    python compress.py --list
    python compress.py --method magnitude --model mobilevit
    python compress.py --method ptq_minmax_w4a8            # a variant config in configs/quant/
    python compress.py --config configs/pipeline/prune_quant.yaml --set hparams.stages.0.hparams.ratio=0.5
    python compress.py --method lsq --set hparams.w_bits=2 hparams.ft_epochs=10
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from mtcompress.config import build_config  # noqa: E402
from mtcompress.registry import KINDS, get_method, list_methods  # noqa: E402


def print_list(kind=None):
    current = None
    for spec in list_methods(kind):
        if spec.kind != current:
            current = spec.kind
            print(f"\n{KINDS[current]}")
        print(f"  {spec.name:<20} {spec.family:<22} {spec.paper}")
    print()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="print every method with family and paper")
    p.add_argument("--kind", choices=list(KINDS), help="with --list: one kind only")
    p.add_argument("--method", help="method name or variant config name")
    p.add_argument("--config", help="YAML config file")
    p.add_argument("--model", help="baseline model (default mobilevit)")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"])
    p.add_argument("--checkpoint", help="baseline weights (default: checkpoints/<model>_<dataset>.pth)")
    p.add_argument("--seed", type=int)
    p.add_argument("--device", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--run-dir", help="default: runs/<variant>_<model>_s<seed>_<time>")
    p.add_argument("--limit-batches", type=int, help="cap batches per epoch and per calibration pass")
    p.add_argument("--no-latency", action="store_true", help="skip PyTorch and ONNX Runtime timing")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides, values as YAML")
    p.add_argument("--hparams", action="store_true", help="print the method's hparams with defaults and exit")
    args = p.parse_args(argv)

    if args.list:
        print_list(args.kind)
        return
    if args.hparams:
        spec = get_method(args.method)
        for k, v in spec.cls.default_hparams().items():
            print(f"{k}: {v}")
        return
    overrides = {"model": args.model, "dataset": args.dataset, "checkpoint": args.checkpoint, "seed": args.seed,
                 "device": args.device, "limit_batches": args.limit_batches}
    if args.no_latency:
        overrides.update({"measure.latency": False, "measure.onnx": False})
    cfg = build_config(args.config, args.method, overrides, args.set)
    from mtcompress.runner import run

    summary = run(cfg, args.run_dir)
    b, f = summary["baseline"], summary["final"]
    print(f"\n{summary['variant']} on {summary['model']}: acc {b['acc']:.2f} -> {f['acc']:.2f} "
          f"({summary['delta_acc']:+.2f}), size {b['size_bytes'] / 1024:.1f} -> {f['size_bytes'] / 1024:.1f} KiB, "
          f"MACs {b['macs'] / 1e6:.2f}M -> {f['macs'] / 1e6:.2f}M")
    if f.get("onnx_latency_ms") and b.get("onnx_latency_ms"):
        print(f"ONNX Runtime CPU latency {b['onnx_latency_ms']:.3f} -> {f['onnx_latency_ms']:.3f} ms")
    print(f"run dir: {summary['run_dir']}")


if __name__ == "__main__":
    main()
