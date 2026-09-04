MECHANISM: Decoupled projection learning rate and weight decay

HYPOTHESIS: Restoring the best verified `sqrt(7/8)` MLP contraction learning rate while preserving the baseline contraction weight-decay magnitude will retain at least 445M training tokens and reduce `val_bpb` below 0.982763.

INTENDED_EDIT: Restore projection-only `sqrt(7/8)` Muon-rate compensation and inversely scale that group’s weight decay so the reduced learning rate does not also weaken cautious decay.

EVIDENCE: Projection-only `sqrt(7/8)` achieved the best observed `val_bpb` of 0.982763, while nearby weaker and stronger reductions regressed; because Muon multiplies both gradient and decay updates by the group learning rate, those trials unintentionally coupled two variables, motivating this controlled separation.

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
            # Keep the best verified contraction update rate, but compensate
            # weight decay because the fused update also multiplies decay by LR.
            is_mlp_projection = shape[1] == 5 * shape[0]
            mlp_lr_scale = math.sqrt(7 / 8) if is_mlp_projection else 1.0
            wd_scale = 1.0 / mlp_lr_scale if is_mlp_projection else 1.0
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr * mlp_lr_scale,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                wd_scale=wd_scale,
            ))
>>>>>>> REPLACE

<<<<<<< SEARCH
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
=======
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay * group["wd_scale"]
>>>>>>> REPLACE