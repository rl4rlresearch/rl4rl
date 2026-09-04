MECHANISM: Equal-area cosine warmdown reshaping

HYPOTHESIS: Replacing linear decay with an equal-area cosine decay will preserve the best schedule’s overall learning-rate budget while annealing more aggressively near the end, achieving `val_bpb < 0.986636`.

INTENDED_EDIT: Keep the 50% warmdown onset and zero final learning rate, but change the warmdown curve from linear to cosine.

EVIDENCE: Warmdown ratios `0.4`, `0.54`, and `0.6` all underperformed the `0.5` design, indicating its duration is near-optimal; changing only the curve tests late-stage annealing without altering architecture, batching, throughput, or integrated learning rate.

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        decay = 0.5 - 0.5 * math.cos(math.pi * cooldown)
        return decay * 1.0 + (1 - decay) * FINAL_LR_FRAC
>>>>>>> REPLACE