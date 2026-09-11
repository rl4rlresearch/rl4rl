MECHANISM: Stronger attention residual-write compensation

HYPOTHESIS: Reducing attention output projections to `7/8` of the base Muon rate while retaining `sqrt(7/8)` for MLP contractions will retain at least 445M training tokens and reduce `val_bpb` below 0.982713.

INTENDED_EDIT: Restore base rates for attention inputs and MLP expansions, retain the verified MLP contraction rate, and move only attention `c_proj` matrices one step lower than the best shared residual-projection rate.

EVIDENCE: Applying `sqrt(7/8)` to both residual-output projection families improved `val_bpb` from the MLP-projection-only result of 0.982763 to 0.982713 at 455.6M tokens, motivating an attention-specific stronger-side rate test while preserving the proven MLP setting.

<<<<<<< SEARCH
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            # The fused Muon step scales LR by sqrt(matrix aspect ratio).
            # Compensate 5x MLP matrices back to the effective rate tuned at 4.375x.
            mlp_lr_scale = math.sqrt(7 / 8) if max(shape) == 5 * min(shape) else 1.0
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr * mlp_lr_scale,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
=======
        attention_projection_ids = {
            id(block.attn.c_proj.weight) for block in self.transformer.h
        }
        mlp_projection_ids = {
            id(block.mlp.c_proj.weight) for block in self.transformer.h
        }
        residual_projection_ids = attention_projection_ids | mlp_projection_ids
        for shape in sorted({p.shape for p in matrix_params}):
            role_groups = (
                ([p for p in matrix_params
                  if p.shape == shape and id(p) not in residual_projection_ids], 1.0),
                ([p for p in matrix_params
                  if p.shape == shape and id(p) in attention_projection_ids], 7 / 8),
                ([p for p in matrix_params
                  if p.shape == shape and id(p) in mlp_projection_ids], math.sqrt(7 / 8)),
            )
            for group_params, projection_lr_scale in role_groups:
                if not group_params:
                    continue
                param_groups.append(dict(
                    kind='muon', params=group_params,
                    lr=matrix_lr * projection_lr_scale,
                    momentum=0.95, ns_steps=5, beta2=0.95,
                    weight_decay=weight_decay,
                ))
>>>>>>> REPLACE