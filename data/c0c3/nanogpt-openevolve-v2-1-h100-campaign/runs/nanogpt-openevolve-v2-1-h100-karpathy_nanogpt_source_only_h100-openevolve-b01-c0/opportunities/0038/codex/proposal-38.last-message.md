MECHANISM: Integral-preserving cosine warmdown

HYPOTHESIS: A cosine-shaped 78.0% warmdown will beat 0.984244 val_bpb by preserving the locally optimal warmdown onset and average learning rate while allocating more learning rate to mid-training and annealing more sharply near the end.

INTENDED_EDIT: Replace the linear warmdown interpolation with a cosine interpolation; retain the 78.0% start point, zero final learning rate, and all other settings.

EVIDENCE: The 78.0% linear warmdown outperformed both adjacent 77.9% and 78.1% settings, so holding its endpoints fixed while changing only schedule curvature is the cleanest orthogonal schedule test.

<<<<<<< SEARCH
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine = 0.5 - 0.5 * math.cos(math.pi * cooldown)
        return cosine * 1.0 + (1 - cosine) * FINAL_LR_FRAC
>>>>>>> REPLACE