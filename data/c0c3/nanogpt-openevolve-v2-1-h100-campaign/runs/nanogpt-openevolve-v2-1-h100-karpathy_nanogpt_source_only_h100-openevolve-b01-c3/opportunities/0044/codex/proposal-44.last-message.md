MECHANISM: Projection-only Muon-rate interpolation

HYPOTHESIS: Applying a fourth-root 7/8 rate reduction to the 512×2560 MLP projections will retain at least 445M training tokens and reduce val_bpb below 0.982763.

INTENDED_EDIT: Restore the proven SSSL attention pattern and reduce only MLP contraction learning rates by `(7/8)^0.25`, halfway in log space between the uncompensated and best verified rates.

EVIDENCE: Projection-only `sqrt(7/8)` achieved the best val_bpb, 0.982763; no reduction reached 0.982905 and the stronger 7/8 reduction regressed to 0.983602, motivating interpolation between the first two rates.

<<<<<<< SEARCH
    window_pattern: str = "SSSS"
=======
    window_pattern: str = "SSSL"
>>>>>>> REPLACE

<<<<<<< SEARCH
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
=======
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            # Interpolate in log space between the uncompensated projection
            # rate and the best verified sqrt(7/8) projection-only rate.
            mlp_lr_scale = (7 / 8) ** 0.25 if shape[1] == 5 * shape[0] else 1.0
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr * mlp_lr_scale,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # final layer is still forced to full context
=======
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
>>>>>>> REPLACE