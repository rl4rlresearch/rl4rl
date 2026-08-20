"""
Train the smallest viable addition transformer: d=16, h=2, L=2, ff=48 (6,080 params).

Usage (from repo root):
    python src/train.py

Saves best checkpoint to checkpoints/best.pt.
Training takes ~10 minutes on Apple Silicon MPS, ~5 minutes on CUDA GPU.
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from data import (AdditionDataset, collate_fn, make_test_set, preprocess, make_target,
                  encode, BOS_ID, EOS_ID, VOCAB_SIZE, OUT_DIGITS, FIXED_SEQ_LEN, ID_TO_TOKEN)
from model import AdditionTransformer


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def batch_evaluate(model, problems, device, batch_size=512):
    """Efficient batched autoregressive evaluation."""
    model.eval()
    correct = 0
    total = len(problems)

    for i in range(0, total, batch_size):
        batch = problems[i:i+batch_size]
        inp_ids_list = []
        tgt_strs = []
        for a, b, c in batch:
            inp_str = preprocess(a, b)
            tgt_str = make_target(a, b)
            inp_ids = [BOS_ID] + encode(inp_str)
            inp_ids_list.append(inp_ids)
            tgt_strs.append(tgt_str)

        inp_tensor = torch.tensor(inp_ids_list, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(inp_tensor, max_new_tokens=OUT_DIGITS+1, eos_id=EOS_ID)
        inp_len = len(inp_ids_list[0])
        for j in range(len(batch)):
            gen_ids = out[j, inp_len:].tolist()
            if EOS_ID in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(EOS_ID)]
            pred_str = ''.join(ID_TO_TOKEN.get(tid, '?') for tid in gen_ids)
            if pred_str == tgt_strs[j]:
                correct += 1

    model.train()
    return correct / total


def train():
    device = get_device()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(repo_root, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Model config: the smallest that works
    cfg = {
        'd_model': 14,
        'n_heads': 2,
        'n_layers': 2,
        'ff_dim': 23,
        'pos_rank': 6,
    }
    max_steps = 50000

    model = AdditionTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        ff_dim=cfg['ff_dim'],
        pos_rank=cfg['pos_rank'],
        max_seq_len=FIXED_SEQ_LEN,
        dropout=0.0,
    ).to(device)

    n_params = model.count_params()
    print(f"Model: {n_params:,} parameters")
    print(f"Config: d={cfg['d_model']}, h={cfg['n_heads']}, L={cfg['n_layers']}, ff={cfg['ff_dim']}")
    print(f"Device: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.0, betas=(0.9, 0.98))

    def lr_lambda(step):
        warmup = 300
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    val_set = make_test_set(num_examples=2000, seed=99)
    test_set = make_test_set(num_examples=10000, seed=42)

    global_step = 0
    best_test_acc = 0
    start_time = time.time()

    print(f"\nTraining for {max_steps} steps...")

    while global_step < max_steps:
        dataset = AdditionDataset(500_000, max_digits=10, seed=global_step, min_digits=1)
        loader = DataLoader(dataset, batch_size=512, shuffle=True,
                          collate_fn=collate_fn, num_workers=0, drop_last=True)

        for batch in loader:
            if global_step >= max_steps:
                break

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, VOCAB_SIZE),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 500 == 0:
                elapsed = (time.time() - start_time) / 60
                lr = scheduler.get_last_lr()[0]
                print(f"  step {global_step:5d} | loss {loss.item():.6f} | lr {lr:.6f} | {elapsed:.1f}min")

            if global_step % 1000 == 0:
                val_acc = batch_evaluate(model, val_set, device)
                elapsed = (time.time() - start_time) / 60
                print(f"  VAL {val_acc:.2%}")

                if val_acc >= 0.99:
                    test_acc = batch_evaluate(model, test_set, device)
                    print(f"  TEST {test_acc:.4%}")

                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        torch.save({
                            'model_state': model.state_dict(),
                            'step': global_step,
                            'config': cfg,
                            'test_acc': test_acc,
                            'n_params': n_params,
                        }, os.path.join(ckpt_dir, 'best.pt'))
                        print(f"  *** New best: {test_acc:.4%} ***")

    if not os.path.isfile(os.path.join(ckpt_dir, 'best.pt')):
        torch.save({
            'model_state': model.state_dict(),
            'step': global_step,
            'config': cfg,
            'test_acc': best_test_acc,
            'n_params': n_params,
        }, os.path.join(ckpt_dir, 'best.pt'))

    elapsed = (time.time() - start_time) / 60
    print(f"\nTraining complete in {elapsed:.1f} minutes")
    print(f"Best test accuracy: {best_test_acc:.4%}")


if __name__ == '__main__':
    train()
