MECHANISM: Cosine learning-rate warmdown

HYPOTHESIS: Replacing the proven 50% linear warmdown with an equal-duration cosine decay will beat 0.985318 val_bpb by preserving average learning-rate exposure while shifting optimization toward higher rates early in cooldown and gentler refinement near completion.

INTENDED_EDIT: Keep the best 262,144-token configuration and 50% warmdown duration, changing only the decay curve from linear to cosine.

EVIDENCE: Shortening linear warmdown to 40% regressed val_bpb from 0.985318 to 0.986825, motivating retention of the 50% duration while testing schedule shape independently.

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine_decay = 0.5 * (1.0 - math.cos(math.pi * cooldown))
        return cosine_decay + (1 - cosine_decay) * FINAL_LR_FRAC
>>>>>>> REPLACE