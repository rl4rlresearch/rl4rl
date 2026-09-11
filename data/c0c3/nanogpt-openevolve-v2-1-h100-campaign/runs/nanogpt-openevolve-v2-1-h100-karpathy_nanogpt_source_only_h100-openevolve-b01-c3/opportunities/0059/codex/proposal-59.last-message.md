MECHANISM: Weaker-side attention residual-rate interpolation

HYPOTHESIS: Retaining `sqrt(7/8)` for MLP contractions while using `(7/8)^0.25` for attention output projections will retain at least 445M training tokens and reduce `val_bpb` below 0.982713.

INTENDED_EDIT: Split Muon groups by parameter identity, restore the best MLP contraction rate, and place attention `c_proj` rates halfway in log space between the baseline and shared-projection optimum.

EVIDENCE: MLP-only compensation reached 0.982763; extending `sqrt(7/8)` to attention outputs improved it to 0.982713, while strengthening attention compensation to `7/8` regressed to 0.982963. This motivates refining the untested weaker side.

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
        matrix_lr_scales = {}
        for block in self.transformer.h:
            # Refine attention on the weaker side of its verified optimum while
            # restoring the best verified MLP contraction compensation.
            matrix_lr_scales[id(block.attn.c_proj.weight)] = (7 / 8) ** 0.25
            matrix_lr_scales[id(block.mlp.c_proj.weight)] = math.sqrt(7 / 8)
        for shape in sorted({p.shape for p in matrix_params}):
            for lr_scale in sorted({
                matrix_lr_scales.get(id(p), 1.0)
                for p in matrix_params if p.shape == shape
            }):
                group_params = [
                    p for p in matrix_params
                    if p.shape == shape and
                    matrix_lr_scales.get(id(p), 1.0) == lr_scale
                ]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=matrix_lr * lr_scale,
                    momentum=0.95, ns_steps=5, beta2=0.95,
                    weight_decay=weight_decay,
                ))
>>>>>>> REPLACE