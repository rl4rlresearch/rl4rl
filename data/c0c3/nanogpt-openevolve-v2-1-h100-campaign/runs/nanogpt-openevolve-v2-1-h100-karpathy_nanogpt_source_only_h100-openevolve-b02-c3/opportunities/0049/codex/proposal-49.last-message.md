MECHANISM: Equal-area cosine warmdown

HYPOTHESIS: Replacing linear warmdown with an equal-area cosine decay will retain at least 490M training tokens and reduce val_bpb below 0.983505 by sustaining useful learning rates longer before decaying more sharply near the end.

INTENDED_EDIT: Preserve the best 212,992-token batch, 23/25 peak learning rates, warmdown duration, and zero terminal rate while changing only the warmdown curve from linear to cosine.

EVIDENCE: At the fixed best batch, peak-rate scales of 9/10, 13/14, and 33/35 all underperformed the current 23/25 result of 0.983505, indicating that peak magnitude is tightly bracketed; an equal-area schedule change tests learning-rate allocation without changing peak or cumulative optimizer displacement.

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine_cooldown = 0.5 - 0.5 * math.cos(math.pi * cooldown)
        return cosine_cooldown + (1 - cosine_cooldown) * FINAL_LR_FRAC
>>>>>>> REPLACE