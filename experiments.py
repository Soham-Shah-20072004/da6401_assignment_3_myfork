"""
experiments.py — W&B report runs (Assignment §2.1–2.5)

Self-contained: imports only stable/contract functions from the skeleton
files and never modifies them. All ablation logic (learned positional
encoding, un-scaled attention, extra logging, attention heat-maps) lives
here.

Run on Kaggle:
    from experiments import run_all, run_one
    run_all()                      # baseline + 5 ablations
    run_one("baseline")            # a single experiment by name
"""

import math
import copy
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import model as model_mod
from model import Transformer, make_src_mask, make_tgt_mask
from train import LabelSmoothingLoss, evaluate_bleu, save_checkpoint
from lr_scheduler import NoamScheduler
from dataset import Multi30kDataset, collate_batch, PAD_IDX, SOS_IDX, EOS_IDX


# ──────────────────────────────────────────────────────────────────────
#  2.4 — Learned positional encoding (swapped onto the model post-build)
# ──────────────────────────────────────────────────────────────────────

class LearnedPositionalEncoding(nn.Module):
    """Drop-in for model.PositionalEncoding: same forward contract."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_emb(pos))


# ──────────────────────────────────────────────────────────────────────
#  2.2 — Un-scaled attention (monkey-patch model.scaled_dot_product_attention)
# ──────────────────────────────────────────────────────────────────────

def _unscaled_attention(Q, K, V, mask=None):
    """Identical to the real fn but WITHOUT the 1/√dₖ scaling."""
    scores = torch.matmul(Q, K.transpose(-2, -1))          # no √dₖ
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    w = F.softmax(scores, dim=-1)
    return torch.matmul(w, V), w


@contextlib.contextmanager
def patched_attention(disable_scaling: bool):
    """Temporarily swap model.scaled_dot_product_attention if requested."""
    if not disable_scaling:
        yield
        return
    original = model_mod.scaled_dot_product_attention
    model_mod.scaled_dot_product_attention = _unscaled_attention
    try:
        yield
    finally:
        model_mod.scaled_dot_product_attention = original


# ──────────────────────────────────────────────────────────────────────
#  Metric helpers
# ──────────────────────────────────────────────────────────────────────

def _qk_grad_norm(model) -> float:
    """L2 norm of gradients of all W_q / W_k weight matrices (expt 2.2)."""
    total = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None and ("W_q" in name or "W_k" in name):
            total += p.grad.detach().norm().item() ** 2
    return total ** 0.5


@torch.no_grad()
def _val_accuracy(model, loader, device) -> float:
    """Token-level next-token accuracy on the validation set (expt 2.1)."""
    model.eval()
    correct, total = 0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tin, tout = tgt[:, :-1], tgt[:, 1:]
        logits = model(src, tin, make_src_mask(src), make_tgt_mask(tin))
        pred = logits.argmax(-1)
        keep = tout != PAD_IDX
        correct += (pred[keep] == tout[keep]).sum().item()
        total += keep.sum().item()
    return correct / max(1, total)


def _confidence(logits, tout) -> float:
    """Mean softmax prob assigned to the GOLD token, non-pad (expt 2.5)."""
    probs = logits.softmax(-1)
    gold = probs.gather(-1, tout.unsqueeze(-1)).squeeze(-1)
    keep = tout != PAD_IDX
    return gold[keep].mean().item()


# ──────────────────────────────────────────────────────────────────────
#  2.3 — Attention heat-maps (last encoder layer, per head)
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def log_encoder_attention(model, src_sentence, src_vocab, nlp_de, device, wandb):
    import matplotlib.pyplot as plt

    model.eval()
    toks = [t.text for t in nlp_de.tokenizer(src_sentence.lower())]
    ids = [SOS_IDX] + src_vocab.encode(toks) + [EOS_IDX]
    src = torch.tensor([ids], device=device)
    model.encode(src, make_src_mask(src))                      # populates .attn

    attn = model.encoder.layers[-1].self_attn.attn[0]          # [h, L, L]
    labels = ["<s>"] + toks + ["</s>"]
    h = attn.size(0)
    fig, axes = plt.subplots(1, h, figsize=(4 * h, 4))
    if h == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(attn[i].cpu(), aspect="auto")
        ax.set_title(f"head {i}")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
    fig.tight_layout()
    wandb.log({"encoder_attention": wandb.Image(fig)})
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
#  Core training routine (one experiment)
# ──────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    name="baseline", project="da6401-a3",
    d_model=256, N=3, num_heads=8, d_ff=1024, dropout=0.1,
    batch_size=128, num_epochs=20, min_freq=2,
    bleu_every=5,              # eval BLEU every N epochs (+ last); BLEU is slow
    scheduler="noam",          # "noam" | "fixed"
    warmup_steps=4000, fixed_lr=1e-4,
    label_smoothing=0.1,       # 2.5: 0.0 to disable
    learned_pos=False,         # 2.4
    unscale_attention=False,   # 2.2
    log_grad_norms=False,      # 2.2
    log_confidence=False,      # 2.5
    viz_attention=False,       # 2.3
    viz_sentence="a man in a blue shirt is standing on a ladder cleaning a window",
)


def _train_one(cfg, datasets):
    import wandb
    cfg = {**DEFAULTS, **cfg}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, test_ds = datasets

    run = wandb.init(project=cfg["project"], name=cfg["name"],
                     group=cfg["name"], config=cfg, reinit=True)

    dl = lambda d, s: DataLoader(d, batch_size=cfg["batch_size"],
                                 shuffle=s, collate_fn=collate_batch)
    train_dl, val_dl, test_dl = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    with patched_attention(cfg["unscale_attention"]):
        model = Transformer(
            src_vocab_size=len(train_ds.src_vocab),
            tgt_vocab_size=len(train_ds.tgt_vocab),
            d_model=cfg["d_model"], N=cfg["N"], num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"], dropout=cfg["dropout"],
        ).to(device)
        if cfg["learned_pos"]:
            model.pos_enc = LearnedPositionalEncoding(
                cfg["d_model"], cfg["dropout"]).to(device)

        if cfg["scheduler"] == "noam":
            optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                         betas=(0.9, 0.98), eps=1e-9)
            scheduler = NoamScheduler(optimizer, cfg["d_model"], cfg["warmup_steps"])
        else:                                   # fixed LR, no warmup (expt 2.1)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["fixed_lr"],
                                         betas=(0.9, 0.98), eps=1e-9)
            scheduler = None

        loss_fn = LabelSmoothingLoss(len(train_ds.tgt_vocab), PAD_IDX,
                                     cfg["label_smoothing"])

        step, best_bleu, last_bleu = 0, 0.0, 0.0
        n_ep = cfg["num_epochs"]
        for epoch in range(n_ep):
            model.train()
            ep_loss, ep_ce, ep_conf, nb = 0.0, 0.0, 0.0, 0
            pbar = tqdm(train_dl, leave=False,
                        desc=f"[{cfg['name']}] epoch {epoch + 1}/{n_ep}")
            for src, tgt in pbar:
                src, tgt = src.to(device), tgt.to(device)
                tin, tout = tgt[:, :-1], tgt[:, 1:]
                logits = model(src, tin, make_src_mask(src), make_tgt_mask(tin))
                flat_logits = logits.reshape(-1, logits.size(-1))
                flat_tgt = tout.reshape(-1)
                loss = loss_fn(flat_logits, flat_tgt)

                optimizer.zero_grad()
                loss.backward()

                log = {"step": step, "lr": optimizer.param_groups[0]["lr"]}
                if cfg["log_grad_norms"] and step < 1000:
                    log["qk_grad_norm"] = _qk_grad_norm(model)
                wandb.log(log)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                # raw (un-smoothed) CE → valid perplexity for §2.5
                with torch.no_grad():
                    ep_ce += F.cross_entropy(
                        flat_logits, flat_tgt, ignore_index=PAD_IDX).item()
                    if cfg["log_confidence"]:
                        ep_conf += _confidence(logits, tout)
                ep_loss += loss.item(); nb += 1; step += 1
                pbar.set_postfix(loss=f"{loss.item():.3f}")

            do_bleu = ((epoch + 1) % cfg["bleu_every"] == 0) or (epoch == n_ep - 1)
            val_acc = _val_accuracy(model, val_dl, device)
            mean_ce = ep_ce / max(1, nb)
            ep_log = {
                "epoch": epoch,
                "train_loss": ep_loss / max(1, nb),
                "train_ppl": math.exp(min(mean_ce, 20)),     # true perplexity
                "val_accuracy": val_acc,
            }
            if cfg["log_confidence"]:
                ep_log["train_confidence"] = ep_conf / max(1, nb)
            if do_bleu:
                last_bleu = evaluate_bleu(model, val_dl, train_ds.tgt_vocab, device)
                ep_log["val_bleu"] = last_bleu
            wandb.log(ep_log)
            print(f"[{cfg['name']}] epoch {epoch + 1}/{n_ep}  "
                  f"train_loss={ep_log['train_loss']:.3f}  "
                  f"ppl={ep_log['train_ppl']:.1f}  "
                  f"val_acc={val_acc:.3f}  "
                  f"val_bleu={last_bleu:.2f}" + ("" if do_bleu else " (stale)"),
                  flush=True)

            if do_bleu and last_bleu >= best_bleu:
                best_bleu = last_bleu
                ckpt = f"checkpoint_{cfg['name']}.pt"
                save_checkpoint(model, optimizer, scheduler, epoch, ckpt)
                art = wandb.Artifact(f"checkpoint-{cfg['name']}", type="model",
                                     metadata={"epoch": epoch, "val_bleu": last_bleu})
                art.add_file(ckpt)
                wandb.log_artifact(art)

        test_bleu = evaluate_bleu(model, test_dl, train_ds.tgt_vocab, device)
        wandb.log({"test_bleu": test_bleu})
        print(f"[{cfg['name']}] best val BLEU={best_bleu:.2f}  test BLEU={test_bleu:.2f}")

        if cfg["viz_attention"]:
            log_encoder_attention(model, cfg["viz_sentence"],
                                   train_ds.src_vocab,
                                   Multi30kDataset._nlp_de, device, wandb)
    run.finish()
    return best_bleu


# ──────────────────────────────────────────────────────────────────────
#  Experiment registry  (Assignment §2.1 – §2.5)
# ──────────────────────────────────────────────────────────────────────

CONFIGS = {
    # baseline also serves §2.3 (attention viz) and the §2.x "with" arms
    "baseline":          dict(name="baseline", viz_attention=True,
                              log_confidence=True),

    # 2.1 — Noam scheduler IS the necessity argument; this is the contrast
    "fixed_lr":          dict(name="fixed_lr", scheduler="fixed",
                              fixed_lr=1e-4),

    # 2.2 — scaling-factor ablation (+ grad-norm logging on both arms)
    "scaled_attn":       dict(name="scaled_attn", log_grad_norms=True),
    "unscaled_attn":     dict(name="unscaled_attn", unscale_attention=True,
                              log_grad_norms=True),

    # 2.4 — learned positional embedding vs sinusoidal (baseline)
    "learned_pos":       dict(name="learned_pos", learned_pos=True),

    # 2.5 — label smoothing off (baseline has eps=0.1, on + confidence)
    "no_label_smooth":   dict(name="no_label_smooth", label_smoothing=0.0,
                              log_confidence=True),
}


def _load_datasets(min_freq=2):
    print("Loading + tokenizing Multi30k (once, shared across experiments)...")
    tr = Multi30kDataset("train", min_freq=min_freq)
    va = Multi30kDataset("validation", tr.src_vocab, tr.tgt_vocab)
    te = Multi30kDataset("test", tr.src_vocab, tr.tgt_vocab)
    return tr, va, te


def run_one(name: str):
    ds = _load_datasets()
    return _train_one(CONFIGS[name], ds)


def run_all():
    ds = _load_datasets()
    results = {}
    for name, cfg in CONFIGS.items():
        results[name] = _train_one(cfg, ds)
    print("\n=== Summary (best val BLEU) ===")
    for k, v in results.items():
        print(f"  {k:18s} {v:.2f}")
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_one(sys.argv[1])
    else:
        run_all()