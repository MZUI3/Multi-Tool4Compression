"""Train an uncompressed baseline: checkpoints/<model>_<dataset>.pth.

    python train.py --model mobilevit --dataset cifar10 --epochs 60
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from mtcompress.common.hwaccel import AccelConfig, Accelerator, seed_everything  # noqa: E402
from mtcompress.data import CifarData  # noqa: E402
from mtcompress.runner import build_model, checkpoint_path  # noqa: E402
from mtcompress.train import evaluate, fit  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="mobilevit")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--epochs", type=float, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=float, default=2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out", default=None, help="default: checkpoints/<model>_<dataset>.pth")
    p.add_argument("--limit-batches", type=int, default=0)
    args = p.parse_args(argv)

    acc = Accelerator(AccelConfig(device=args.device, amp=args.amp), log=print)
    seed_everything(args.seed, rank_offset=False)
    data = CifarData(args.dataset, args.data_root, acc.device, batch_size=args.batch_size, seed=args.seed)
    model = build_model(args.model, data.num_classes).to(acc.device)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters())} parameters, {acc.summary()}")
    t0 = time.time()
    hist = fit(model, data, args.epochs, args.lr, optimizer=args.optimizer, weight_decay=args.weight_decay,
               warmup_epochs=args.warmup_epochs, label_smoothing=args.label_smoothing, batch_size=args.batch_size,
               amp=acc.amp, limit_batches=args.limit_batches, eval_every=5, tag="baseline")
    result = evaluate(model, data.val(), acc.amp)
    out = args.out or checkpoint_path(args.model, args.dataset)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = {"model": args.model, "dataset": args.dataset, "val_acc": result["acc"], "epochs": args.epochs,
            "args": vars(args), "train_time_s": round(time.time() - t0, 1), "torch": str(torch.__version__)}
    torch.save({"model_state": model.state_dict(), "meta": meta}, out)
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({**meta, "history": hist}, f, indent=1)
    print(f"saved {out}: val acc {result['acc']:.2f}%")


if __name__ == "__main__":
    main()
