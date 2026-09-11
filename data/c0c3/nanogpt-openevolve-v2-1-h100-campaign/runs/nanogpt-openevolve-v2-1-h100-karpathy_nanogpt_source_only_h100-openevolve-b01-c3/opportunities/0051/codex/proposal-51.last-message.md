MECHANISM: Projection-specific NorMuon variance smoothing

HYPOTHESIS: Restoring the best `sqrt(7/8)` contraction rate and doubling only contraction matrices’ variance-estimation horizon will retain at least 445M training tokens and reduce `val_bpb` below 0.982763.

INTENDED_EDIT: Use the verified-best learning rate for 512×2560 MLP projections and set their NorMuon `beta2` to `sqrt(0.95)`, leaving all other groups unchanged.

EVIDENCE: Projection-only `sqrt(7/8)` achieved the best observed `val_bpb` of 0.982763, while nearby learning-rate and weight-decay refinements regressed; projection-specific variance smoothing tests an independent stabilization axis without adding model computation.

<<<<<<< SEARCH
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            # Interpolate in log space between the uncompensated projection
            # rate and the best verified sqrt(7/8) projection-only rate.
            mlp_lr_scale = (7 / 8) ** 0.25 if shape[1] == 5 * shape[0] else 1.0
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr * mlp_lr_scale,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
=======
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            is_mlp_projection = shape[1] == 5 * shape[0]
            projection_lr_scale = math.sqrt(7 / 8) if is_mlp_projection else 1.0
            # Double the variance-estimation horizon only for MLP contractions.
            beta2 = math.sqrt(0.95) if is_mlp_projection else 0.95
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr * projection_lr_scale,
                momentum=0.95, ns_steps=5, beta2=beta2, weight_decay=weight_decay,
            ))
>>>>>>> REPLACE