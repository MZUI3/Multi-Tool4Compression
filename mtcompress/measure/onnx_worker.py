"""ONNX export and ONNX Runtime timing in a child process.

The TorchScript ONNX exporter can crash the interpreter outright on some graphs (heap corruption while
optimising ToMe's scatter_reduce), which no ``try`` can catch. ``run_isolated`` pickles the model and the
calibration batch to a temporary folder, runs this module on them and reads the JSON it writes, so a
crash costs one measurement, not the benchmark.

    python -m mtcompress.measure.onnx_worker <folder>
"""
import json
import os
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_isolated(model, example, calib, threads: int, fp32: bool, int8: bool, timeout: int = 900) -> dict:
    folder = tempfile.mkdtemp(prefix="mtc_onnx_")
    torch.save({"model": model.cpu(), "example": example.cpu(), "calib": [c.cpu() for c in calib],
                "threads": threads, "fp32": fp32, "int8": int8}, os.path.join(folder, "job.pt"))
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        proc = subprocess.run([sys.executable, "-m", "mtcompress.measure.onnx_worker", folder], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ONNX measurement timed out after {timeout}s")
    out_path = os.path.join(folder, "result.json")
    if proc.returncode != 0 or not os.path.isfile(out_path):
        tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
        raise RuntimeError(f"ONNX worker exited with code {proc.returncode}: {tail[0][:200]}")
    with open(out_path, encoding="utf-8") as f:
        return json.load(f)


def main(folder: str) -> None:
    import warnings

    warnings.filterwarnings("ignore")
    from mtcompress.measure.profile import onnx_latencies

    job = torch.load(os.path.join(folder, "job.pt"), weights_only=False)
    result = onnx_latencies(job["model"], job["example"], job["calib"], job["threads"], fp32=job["fp32"],
                            int8=job["int8"])
    with open(os.path.join(folder, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main(sys.argv[1])
