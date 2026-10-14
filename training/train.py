"""Small GPT-style training job you can break on purpose.

Everything is controlled with environment variables so you can change one thing at a time:
  BATCH        samples per step per GPU                         (default 16)
  NUM_WORKERS  DataLoader worker processes                       (default 4)
  DATA_MS      CPU milliseconds spent preparing each sample      (default 0)
               (stands in for decoding, tokenizing, augmenting)
  STEPS        total steps                                       (default 3000)
  CKPT_DIR     directory for checkpoints; empty disables them    (default "")
  CKPT_EVERY   save a checkpoint every N steps                   (default 0)
  NCCL_TIMEOUT seconds before a stuck collective is aborted      (default 600)
Runs on one GPU with `python train.py`, or across nodes with torchrun (DDP).
"""
import datetime
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROC_START = time.time()


def env(name, default):
    return type(default)(os.environ.get(name, default))


BATCH = env("BATCH", 16)
SEQ = env("SEQ", 512)
DIM = env("DIM", 1024)
LAYERS = env("LAYERS", 12)
VOCAB = 32000
WORKERS = env("NUM_WORKERS", 4)
DATA_MS = env("DATA_MS", 0.0)
STEPS = env("STEPS", 3000)
CKPT_DIR = os.environ.get("CKPT_DIR", "")
CKPT_EVERY = env("CKPT_EVERY", 0)
NCCL_TIMEOUT = env("NCCL_TIMEOUT", 600)
LOG_EVERY = env("LOG_EVERY", 20)


class SyntheticTokens(Dataset):
    """Random token sequences. DATA_MS burns real CPU time per sample."""

    def __len__(self):
        return 10**9

    def __getitem__(self, i):
        if DATA_MS > 0:
            end = time.perf_counter() + DATA_MS / 1000
            while time.perf_counter() < end:
                pass
        x = torch.randint(0, VOCAB, (SEQ + 1,))
        return x[:-1], x[1:]


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        layer = nn.TransformerEncoderLayer(DIM, DIM // 64, 4 * DIM, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        return self.head(self.body(self.emb(x), mask=mask, is_causal=True))


def main():
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=NCCL_TIMEOUT))
    rank = dist.get_rank() if ddp else 0
    world = dist.get_world_size() if ddp else 1
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    # So you can freeze this exact process from outside: kill -STOP $(cat /tmp/train.pid)
    with open("/tmp/train.pid", "w") as f:
        f.write(str(os.getpid()))

    model = TinyGPT().to(dev)
    raw = model
    if ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    start = 0
    ckpt = os.path.join(CKPT_DIR, "ckpt.pt") if CKPT_DIR else None
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=dev)
        raw.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start = state["step"]
        print(f"[rank {rank}] resumed from checkpoint at step {start}", flush=True)

    if rank == 0:
        n = sum(p.numel() for p in raw.parameters())
        print(f"params={n/1e6:.0f}M batch={BATCH} seq={SEQ} workers={WORKERS} "
              f"data_ms={DATA_MS} world={world} gpu={torch.cuda.get_device_name(dev)}", flush=True)

    loader = iter(DataLoader(SyntheticTokens(), batch_size=BATCH, num_workers=WORKERS,
                             pin_memory=True, persistent_workers=WORKERS > 0))
    last = time.time()
    for step in range(start + 1, STEPS + 1):
        x, y = next(loader)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == start + 1 and rank == 0:
            print(f"first step done {time.time() - PROC_START:.0f} s after the process started", flush=True)
        if step % LOG_EVERY == 0:
            torch.cuda.synchronize()
            now = time.time()
            dt = (now - last) / LOG_EVERY
            last = now
            if rank == 0:
                print(f"step {step:5d}  loss {loss.item():.3f}  step_time {dt*1000:6.0f} ms  "
                      f"samples/s {BATCH*world/dt:6.0f}  "
                      f"mem_peak {torch.cuda.max_memory_allocated(dev)/2**30:5.1f} GiB", flush=True)
        if ckpt and CKPT_EVERY and step % CKPT_EVERY == 0 and rank == 0:
            t = time.time()
            torch.save({"model": raw.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)  # atomic: a crash mid-save never leaves a broken checkpoint
            print(f"checkpoint saved at step {step} in {time.time() - t:.1f} s", flush=True)

    if rank == 0:
        print("training finished", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
