MECHANISM: Equal-area cosine terminal annealing

HYPOTHESIS: Replacing the linear warmdown with an equal-area cosine warmdown at the proven 93.75% peak learning rates will retain at least 460M-token throughput and reduce val_bpb below 0.984614.

INTENDED_EDIT: Keep the architecture, batch size, peak learning rates, warmdown duration, endpoints, and cumulative learning-rate exposure fixed while changing only the warmdown curvature.

EVIDENCE: The 262,144-token runs bracket the best peak scale: 93.75% achieved 0.984614, while 87.5%, 100%, and 81.25% were worse. Holding that peak fixed and testing an equal-area schedule shape is therefore a controlled next optimization dimension.

<<<<<<< SEARCH
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine_decay = 0.5 * (1.0 - math.cos(math.pi * cooldown))
        return cosine_decay + (1.0 - cosine_decay) * FINAL_LR_FRAC
>>>>>>> REPLACE