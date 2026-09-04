MECHANISM: Equal-area cosine warmdown

HYPOTHESIS: Restoring the best projection-only `sqrt(7/8)` Muon rate and replacing linear warmdown with an equal-area cosine curve will retain at least 445M tokens and reduce `val_bpb` below 0.982763 by suppressing disruptive terminal updates without reducing the total learning-rate budget.

INTENDED_EDIT: Restore baseline scheduled weight decay for every Muon group, retain contraction-only learning-rate compensation, and use cosine rather than linear decay over the final half of training.

EVIDENCE: Projection-only `sqrt(7/8)` compensation achieved the best verified `val_bpb` of 0.982763 at 450.6M tokens; later projection optimizer refinements regressed, while the independent equal-area cosine schedule attempt produced no verifiable implementation and therefore remains untested.

<<<<<<< SEARCH
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
=======
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            is_mlp_projection = shape[1] == 5 * shape[0]
            projection_scale = math.sqrt(7 / 8) if is_mlp_projection else 1.0
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr * projection_scale,
                momentum=0.95, ns_steps=5, beta2=0.95,
                weight_decay=weight_decay,
            ))
>>>>>>> REPLACE

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine = 0.5 - 0.5 * math.cos(math.pi * cooldown)
        return cosine * 1.0 + (1 - cosine) * FINAL_LR_FRAC
>>>>>>> REPLACE

<<<<<<< SEARCH
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay * group["weight_decay_scale"]
=======
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
>>>>>>> REPLACE