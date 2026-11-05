"""Iterative magnitude pruning (IMP) with rewinding.

Each round trains for ``round_epochs``, removes ``rate`` of the remaining weights by global magnitude and
then rewinds:

- ``rewind: weights``: back to the weights the run started from (the lottery ticket procedure; the
  starting point is the trained model, or a random init with ``from_scratch: true``).
- ``rewind: lr``: keep the weights and restart the learning-rate schedule (learning-rate rewinding).
- ``rewind: none``: keep weights, continue with a short fine-tune (iterative pruning of Han et al., 2015).

After the last pruning step the network is trained once more for ``ft_epochs``.
"""
import copy

from ..base import PruneMethod
from ..registry import register_method
from .masks import MaskSet


@register_method("imp", kind="prune", family="Lottery ticket",
                 paper="Frankle & Carbin, 2019; Renda et al., 2020 (LR rewinding)")
class IMP(PruneMethod):
    hparams = {"rounds": 4, "rate": 0.4, "rewind": "lr", "round_epochs": 3, "from_scratch": False,
               "ft_epochs": 3}

    def apply(self, ctx):
        hp = self.hp
        if hp["rewind"] not in ("weights", "lr", "none"):
            raise ValueError("rewind must be weights, lr or none")
        if hp["from_scratch"]:
            ctx.model = ctx.build_model()
        masks = MaskSet(ctx.model, hp["exclude"])
        start = copy.deepcopy(ctx.model.state_dict())
        rounds = []
        for r in range(hp["rounds"]):
            if hp["from_scratch"] or r > 0 or hp["rewind"] != "weights":
                self.finetune(ctx, hooks=masks, epochs=hp["round_epochs"], tag=f"imp/round{r}")
            target = 1 - (1 - hp["rate"]) ** (r + 1)
            masks.prune_by_scores(masks.scores(lambda n, w: w.detach().abs()), target, "global")
            acc = ctx.evaluate()["acc"]
            rounds.append({"round": r, "sparsity": round(100 * masks.sparsity(), 2), "acc_before_retrain": acc})
            ctx.log(f"imp round {r}: sparsity {100 * masks.sparsity():.1f}% acc {acc:.2f}")
            if hp["rewind"] == "weights":
                ctx.model.load_state_dict(start)
                masks.apply()
        self.finetune(ctx, hooks=masks, tag="imp/final")
        ctx.info["rounds"] = rounds
        masks.finalize(ctx.model, ctx.info)
