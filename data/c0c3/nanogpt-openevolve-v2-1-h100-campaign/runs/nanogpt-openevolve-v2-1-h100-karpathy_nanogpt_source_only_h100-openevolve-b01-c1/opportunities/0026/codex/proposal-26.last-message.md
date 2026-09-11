MECHANISM: Equal-area cosine warmdown

HYPOTHESIS: Replacing linear warmdown with an equal-area cosine curve will shift learning-rate mass from noisier late updates to earlier warmdown updates and lower val_bpb below 0.985229.

INTENDED_EDIT: Keep the successful 0.625 warmdown duration and all other settings unchanged, but use cosine interpolation between peak and final learning rates.

EVIDENCE: Linear WARMDOWN_RATIO 0.625 achieved 0.985229, while extending it to 0.7 regressed to 0.985706; reshaping the best-duration schedule preserves its average learning rate while reducing late-step magnitude.

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