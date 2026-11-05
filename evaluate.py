"""Re-measure a saved run: accuracy on the test split and the cost numbers.

    python evaluate.py runs/magnitude_mobilevit_s0_20261106-120000
    python evaluate.py runs/ptq_minmax_mobilevit_s0_* --no-latency
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from mtcompress.config import DEFAULTS  # noqa: E402
from mtcompress.data import CifarData  # noqa: E402
from mtcompress.runner import load_run, profile  # noqa: E402
from mtcompress.train import evaluate  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--no-latency", action="store_true")
    args = p.parse_args(argv)
    import torch

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    dirs = [d for pattern in args.run_dirs for d in sorted(glob.glob(pattern))]
    data = None
    for d in dirs:
        model, cfg, summary = load_run(d, device)
        if data is None or data.name != cfg["dataset"]:
            data = CifarData(cfg["dataset"], args.data_root, device)
        result = evaluate(model, data.val())
        measure = dict(DEFAULTS["measure"])
        if args.no_latency:
            measure.update(latency=False, onnx=False)
        prof = profile(model, data, measure)
        saved = summary["final"]["acc"]
        print(f"{os.path.basename(d)}: acc {result['acc']:.2f}% (saved {saved:.2f}%), "
              f"size {prof['size_bytes'] / 1024:.1f} KiB, MACs {prof['macs'] / 1e6:.2f}M, "
              f"onnx {prof.get('onnx_latency_ms')} ms")
        print(json.dumps({k: v for k, v in prof.items() if not isinstance(v, dict)}, indent=None, default=str))


if __name__ == "__main__":
    main()
