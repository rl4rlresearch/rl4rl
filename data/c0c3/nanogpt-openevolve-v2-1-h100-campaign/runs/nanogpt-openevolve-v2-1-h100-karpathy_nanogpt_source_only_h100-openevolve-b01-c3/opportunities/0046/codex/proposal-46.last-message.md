MECHANISM: Stronger-side projection Muon-rate interpolation

HYPOTHESIS: Setting only the 512×2560 MLP contraction rate to `(7/8)^0.75` will retain at least 445M training tokens and reduce `val_bpb` below 0.982763.

INTENDED_EDIT: Restore the expansion and attention matrices to the base Muon rate, while moving the contraction rate halfway in log space between the best verified `sqrt(7/8)` rate and the over-reduced `7/8` rate.

EVIDENCE: Projection-only `sqrt(7/8)` achieved the best observed `val_bpb` of 0.982763, while `7/8` regressed to 0.983602; less-reduced fourth-root and three-eighths rates also failed to improve, so the untested stronger-side midpoint is the most informative local refinement.

<<<<<<< SEARCH
            # The fused Muon step scales LR by sqrt(matrix aspect ratio).
            # Compensate 5x MLP matrices back to the effective rate tuned at 4.375x.
            mlp_lr_scale = math.sqrt(7 / 8) if max(shape) == 5 * min(shape) else 1.0
=======
            # Refine the stronger side of the best projection-only rate:
            # halfway in log space between sqrt(7/8) and 7/8.
            mlp_lr_scale = (7 / 8) ** 0.75 if shape[1] == 5 * shape[0] else 1.0
>>>>>>> REPLACE