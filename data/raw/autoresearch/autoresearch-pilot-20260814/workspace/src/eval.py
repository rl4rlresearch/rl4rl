"""
Evaluate the trained addition transformer on held-out test problems.

Usage (from repo root):
    python src/eval.py                          # evaluate checkpoint
    python src/eval.py --inference 123 456      # single inference: 123 + 456

Loads checkpoints/best.pt, evaluates on 10,000 test problems (seed=42),
and saves detailed results to results/final_results.json.
"""

import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from data import (make_test_set, preprocess, make_target, postprocess,
                  encode, ID_TO_TOKEN, VOCAB_SIZE, BOS_ID, EOS_ID,
                  OUT_DIGITS, FIXED_SEQ_LEN)
from model import AdditionTransformer


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_model(ckpt_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']

    model = AdditionTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=cfg.get('d_model', 16),
        n_heads=cfg.get('n_heads', 2),
        n_layers=cfg.get('n_layers', 2),
        ff_dim=cfg.get('ff_dim', 48),
        max_seq_len=FIXED_SEQ_LEN,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    n_params = model.count_params()
    print(f"Loaded model: {n_params:,} parameters")
    print(f"Config: d={cfg['d_model']}, h={cfg['n_heads']}, L={cfg['n_layers']}, ff={cfg['ff_dim']}")
    print(f"Checkpoint step: {ckpt.get('step', 'N/A')}, test_acc: {ckpt.get('test_acc', 'N/A')}")
    return model


def inference(model, a, b, device):
    """Run a single addition through the model."""
    inp_str = preprocess(a, b)
    inp_ids = [BOS_ID] + encode(inp_str)
    inp_tensor = torch.tensor([inp_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(inp_tensor, max_new_tokens=OUT_DIGITS+1, eos_id=EOS_ID)

    gen_ids = out[0, len(inp_ids):].tolist()
    if EOS_ID in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(EOS_ID)]
    raw_output = ''.join(ID_TO_TOKEN.get(i, '?') for i in gen_ids)
    predicted = postprocess(raw_output)
    return predicted, raw_output


def full_evaluation(model, device, results_path=None):
    """Run comprehensive evaluation on 10,000 test problems."""
    print("\n" + "="*60)
    print("EVALUATION ON 10,000 TEST PROBLEMS (seed=42)")
    print("="*60)

    test_set = make_test_set(num_examples=10000, seed=42)

    correct = 0
    failures = []
    per_pos_correct = [0] * OUT_DIGITS
    per_pos_total = [0] * OUT_DIGITS
    per_ndigits_correct = {}
    per_ndigits_total = {}

    batch_size = 512
    total = len(test_set)

    for i in range(0, total, batch_size):
        batch = test_set[i:i+batch_size]
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
            a, b, c = batch[j]
            gen_ids = out[j, inp_len:].tolist()
            if EOS_ID in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(EOS_ID)]
            pred_str = ''.join(ID_TO_TOKEN.get(tid, '?') for tid in gen_ids)
            tgt_str = tgt_strs[j]

            n_answer_digits = len(str(c))
            per_ndigits_total.setdefault(n_answer_digits, 0)
            per_ndigits_correct.setdefault(n_answer_digits, 0)
            per_ndigits_total[n_answer_digits] += 1

            if pred_str == tgt_str:
                correct += 1
                per_ndigits_correct[n_answer_digits] += 1
            else:
                pred_val = postprocess(pred_str) if pred_str else -1
                failures.append({
                    'a': a, 'b': b, 'expected': c,
                    'predicted': pred_val,
                    'pred_str': pred_str,
                    'tgt_str': tgt_str,
                })

            for pos in range(OUT_DIGITS):
                per_pos_total[pos] += 1
                if pos < len(pred_str) and pos < len(tgt_str) and pred_str[pos] == tgt_str[pos]:
                    per_pos_correct[pos] += 1

    acc = correct / total
    print(f"\nExact-match accuracy: {correct}/{total} = {acc:.4%}")

    print(f"\nPer-position accuracy:")
    for pos in range(OUT_DIGITS):
        if per_pos_total[pos] > 0:
            pos_acc = per_pos_correct[pos] / per_pos_total[pos]
            print(f"  Position {pos:2d}: {pos_acc:.4%}")

    print(f"\nPer answer-digit-count accuracy:")
    for nd in sorted(per_ndigits_total.keys()):
        nd_acc = per_ndigits_correct[nd] / per_ndigits_total[nd]
        print(f"  {nd:2d}-digit answers: {per_ndigits_correct[nd]}/{per_ndigits_total[nd]} = {nd_acc:.2%}")

    if failures:
        print(f"\nFailure cases ({len(failures)} total):")
        for f in failures[:20]:
            print(f"  {f['a']} + {f['b']} = {f['expected']}, predicted: {f['predicted']}")
    else:
        print("\nNo failures!")

    results = {
        'exact_match_accuracy': acc,
        'correct': correct,
        'total': total,
        'per_position_accuracy': [per_pos_correct[i]/per_pos_total[i] if per_pos_total[i] > 0 else 0 for i in range(OUT_DIGITS)],
        'per_ndigits_accuracy': {str(k): per_ndigits_correct[k]/per_ndigits_total[k] for k in sorted(per_ndigits_total.keys())},
        'num_failures': len(failures),
        'failures': failures[:50],
    }

    if results_path:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate addition transformer')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: checkpoints/best.pt)')
    parser.add_argument('--inference', nargs=2, type=int, metavar=('A', 'B'),
                        help='Run single inference: A + B')
    args = parser.parse_args()

    device = get_device()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = args.checkpoint or os.path.join(repo_root, 'checkpoints', 'best.pt')

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Train first with: python src/train.py")
        sys.exit(1)

    model = load_model(ckpt_path, device)

    if args.inference:
        a, b = args.inference
        predicted, raw = inference(model, a, b, device)
        expected = a + b
        status = "CORRECT" if predicted == expected else "WRONG"
        print(f"\n{a} + {b} = {predicted}  (expected: {expected})  [{status}]")
        print(f"Raw model output (reversed): {raw}")
    else:
        results_path = os.path.join(repo_root, 'results', 'final_results.json')
        full_evaluation(model, device, results_path)
