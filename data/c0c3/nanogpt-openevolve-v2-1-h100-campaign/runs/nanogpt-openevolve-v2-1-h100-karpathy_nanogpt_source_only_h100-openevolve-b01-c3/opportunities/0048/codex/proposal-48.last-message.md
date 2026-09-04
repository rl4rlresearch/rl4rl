MECHANISM: Stronger projection-only cautious-decay reduction

HYPOTHESIS: Restoring the best 5× MLP and `sqrt(7/8)` contraction learning rate while applying a second `sqrt(7/8)` reduction to contraction weight decay will retain at least 445M tokens and reduce `val_bpb` below 0.982763.

INTENDED_EDIT: Restore 2560-channel MLPs, apply the verified-best projection-only learning-rate compensation, and reduce only those projections’ scheduled weight decay so their effective decay update is 7/8 of baseline.

EVIDENCE: Projection-only `sqrt(7/8)` learning-rate compensation achieved the best `val_bpb` of 0.982763, while restoring the projection’s baseline effective decay regressed to 0.982957, motivating a controlled move toward weaker projection decay.

<<<<<<< SEARCH
        hidden_dim = 21 * config.n_embd // 4
=======
        hidden_dim = 5 * config.n_embd
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
            is_mlp_projection = shape[1] == 5 * shape[0]
            projection_scale = math.sqrt(7 / 8) if is_mlp_projection else 1.0
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr * projection_scale,
                momentum=0.95, ns_steps=5, beta2=0.95,
                weight_decay=weight_decay * projection_scale,
                weight_decay_scale=projection_scale,
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