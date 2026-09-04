MECHANISM: Selective Polar Express step reduction

HYPOTHESIS: Using four orthogonalization steps for non-projection matrices while preserving five steps and the verified-best rate for MLP contractions will retain at least 450M training tokens and reduce `val_bpb` below 0.982763.

INTENDED_EDIT: Keep the sensitive 512×2560 MLP projections unchanged, but reduce Muon’s Polar Express iterations from five to four for attention and MLP expansion matrices to trade marginal orthogonalization accuracy for higher fixed-time throughput.

EVIDENCE: Projection-only `sqrt(7/8)` compensation achieved the best result at 0.982763 and 450.6M tokens, while subsequent projection learning-rate, decay, and variance refinements regressed; this preserves that projection configuration and tests a separate throughput axis.

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
            # Preserve the verified-best contraction update, while using one
            # fewer Polar Express iteration for less-sensitive matrix groups.
            is_mlp_projection = shape[1] == 5 * shape[0]
            mlp_lr_scale = math.sqrt(7 / 8) if is_mlp_projection else 1.0
            ns_steps = 5 if is_mlp_projection else 4
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr * mlp_lr_scale,
                momentum=0.95, ns_steps=ns_steps, beta2=0.95, weight_decay=weight_decay,
            ))
>>>>>>> REPLACE