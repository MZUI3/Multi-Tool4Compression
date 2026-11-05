"""Every method runs end to end on a randomly initialised model with two batches per pass.

This checks that a method runs, saves and reloads, not that it compresses well.

    python tests/test_smoke.py                # all methods
    python tests/test_smoke.py lsq brecq      # a subset
    pytest tests/
"""
import os
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mtcompress.config import build_config  # noqa: E402
from mtcompress.registry import list_methods  # noqa: E402

# Keep the expensive loops short.
FAST = {
    "iters": 10, "iterations": 5, "rounds": 2, "round_epochs": 1, "hutchinson_samples": 2, "samples": 256,
    "calib_batches": 2, "score_batches": 2, "sparsity_epochs": 1, "measure_gpu": False,
}
VIT_ONLY = {"tome", "evit", "fq_vit", "ptq4vit"}


def run_one(name: str, model: str = None, device: str = "auto") -> dict:
    from mtcompress.registry import get_method
    from mtcompress.runner import load_run, run

    spec = get_method(name)
    defaults = spec.cls.default_hparams()
    hp = {k: v for k, v in FAST.items() if k in defaults}
    if "ft_epochs" in defaults:
        hp["ft_epochs"] = 1
    if spec.kind == "pipeline":
        stages = []
        example = [{"method": "magnitude"}, {"method": "ptq_minmax"}]
        for st in defaults["stages"] or example:
            sd = get_method(st["method"]).cls.default_hparams()
            shp = {**st.get("hparams", {}), **{k: v for k, v in FAST.items() if k in sd}, "ft_epochs": 1}
            stages.append({"method": st["method"], "hparams": shp})
        hp = {"stages": stages}
    model = model or ("vit_tiny" if name in VIT_ONLY else "mobilevit")
    cfg = build_config(None, name, {"model": model, "checkpoint": "none", "limit_batches": 2, "device": device,
                                    "save_dir": tempfile.mkdtemp()})
    cfg["hparams"] = {**cfg.get("hparams", {}), **hp}
    summary = run(cfg, log=lambda *a, **k: None)
    model_obj, _, _ = load_run(summary["run_dir"])
    assert model_obj is not None
    return summary


def test_all_methods():
    failures = []
    for spec in list_methods():
        try:
            run_one(spec.name)
        except Exception:
            failures.append((spec.name, traceback.format_exc()))
    assert not failures, "\n".join(f"{n}:\n{t}" for n, t in failures)


def main(names):
    names = names or [s.name for s in list_methods()]
    ok, bad = 0, []
    for name in names:
        model = None
        if ":" in name:
            name, model = name.split(":", 1)
        try:
            s = run_one(name, model)
            ok += 1
            f = s["final"]
            print(f"ok   {name:<20} acc {f['acc']:6.2f}  params {f['params']:>9}  nnz {f['weights_nonzero']:>9}  "
                  f"size {f['size_bytes'] / 1024:8.1f} KiB  MACs {f['macs'] / 1e6:7.2f}M")
        except Exception as exc:
            bad.append(name)
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{ok} passed, {len(bad)} failed {bad if bad else ''}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
