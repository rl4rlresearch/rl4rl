MECHANISM: Equal-area cosine warmdown with projection-only Muon compensation

HYPOTHESIS: Restoring the verified-best projection-only `sqrt(7/8)` rate and using an equal-area cosine warmdown will retain at least 445M tokens and reduce `val_bpb` below 0.982763 by reducing disruptive terminal updates without lowering the total scheduled learning-rate budget.

INTENDED_EDIT: Leave MLP expansion and attention rates unchanged, compensate only MLP contractions, and replace linear LR warmdown with a cosine curve having identical endpoints and mean multiplier.

EVIDENCE: Projection-only compensation achieved the best result, 0.982763 at 450.6M tokens, while subsequent rate, decay, variance, and orthogonalization refinements regressed; this preserves that configuration and tests an independent, throughput-neutral scheduling axis.

<<<<<<< SEARCH
            # The fused Muon step scales LR by sqrt(matrix aspect ratio).
            # Compensate 5x MLP matrices back to the effective rate tuned at 4.375x.
            mlp_lr_scale = math.sqrt(7 / 8) if max(shape) == 5 * min(shape) else 1.0
=======
            # Preserve the base rate for expansions and compensate only the
            # 512x2560 MLP contraction matrices.
            mlp_lr_scale = math.sqrt(7 / 8) if shape[1] == 5 * shape[0] else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine = 0.5 * (1 - math.cos(math.pi * cooldown))
        return cosine * 1.0 + (1 - cosine) * FINAL_LR_FRAC
>>>>>>> REPLACE