MECHANISM: Projection cautious-decay log-space interpolation

HYPOTHESIS: Increasing only the 512×2560 MLP projection weight decay by `(8/7)^0.25` while retaining its verified-best `sqrt(7/8)` learning rate will retain at least 445M training tokens and reduce `val_bpb` below 0.982763.

INTENDED_EDIT: Move the projection’s effective cautious-decay magnitude halfway in log space from the current best toward baseline, without changing its learning rate or other parameter groups.

EVIDENCE: The current effective projection-decay factor achieved 0.982763, while reducing it to 7/8 regressed to 0.983097 and restoring it to 1.0 regressed less severely to 0.982957; this brackets the optimum and motivates refinement on the better upward side.

<<<<<<< SEARCH
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            # Isolate the component favored by prior ablations: lower the rate
            # only for 512x2560 MLP contraction weights.
            mlp_lr_scale = math.sqrt(7 / 8) if shape[1] == 5 * shape[0] else 1.0
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr * mlp_lr_scale,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
=======
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            is_mlp_projection = shape[1] == 5 * shape[0]
            projection_lr_scale = math.sqrt(7 / 8) if is_mlp_projection else 1.0
            # With the projection LR fixed at sqrt(7/8), this moves its effective
            # decay halfway in log space from sqrt(7/8) toward baseline.
            weight_decay_scale = (8 / 7) ** 0.25 if is_mlp_projection else 1.0
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr * projection_lr_scale,
                momentum=0.95, ns_steps=5, beta2=0.95,
                weight_decay=weight_decay * weight_decay_scale,
                weight_decay_scale=weight_decay_scale,
            ))
>>>>>>> REPLACE

<<<<<<< SEARCH
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
=======
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay * group["weight_decay_scale"]
>>>>>>> REPLACE