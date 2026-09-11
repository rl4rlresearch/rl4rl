MECHANISM: Residual-output projection Muon-rate compensation

HYPOTHESIS: Applying the verified `sqrt(7/8)` Muon-rate reduction to both zero-initialized MLP and attention output projections will retain at least 445M training tokens and reduce `val_bpb` below 0.982763.

INTENDED_EDIT: Split transformer matrices by residual-output role; retain base rates for attention input projections and MLP expansions while reducing both `c_proj` families by `sqrt(7/8)`.

EVIDENCE: Contraction-only compensation produced the best result, 0.982763 at 450.6M tokens, while nearby rate, decay, and variance refinements regressed. Attention `c_proj` shares the contraction’s zero initialization and residual-write role, making it the closest untested extension without adding forward computation.

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
        residual_projection_ids = {
            id(proj.weight)
            for block in self.transformer.h
            for proj in (block.attn.c_proj, block.mlp.c_proj)
        }
        for shape in sorted({p.shape for p in matrix_params}):
            for is_residual_projection in (False, True):
                group_params = [
                    p for p in matrix_params
                    if p.shape == shape and
                    (id(p) in residual_projection_ids) == is_residual_projection
                ]
                if not group_params:
                    continue
                # Both output projections are zero-initialized residual writes;
                # extend the verified MLP-contraction compensation to attention.
                projection_lr_scale = math.sqrt(7 / 8) if is_residual_projection else 1.0
                param_groups.append(dict(
                    kind='muon', params=group_params,
                    lr=matrix_lr * projection_lr_scale,
                    momentum=0.95, ns_steps=5, beta2=0.95,
                    weight_decay=weight_decay,
                ))
>>>>>>> REPLACE