MECHANISM: Projection-specific short-horizon NorMuon adaptation

HYPOTHESIS: On the restored best projection-only learning-rate and decay configuration, reducing contraction-matrix NorMuon `beta2` to 0.90 will retain at least 445M tokens and reduce `val_bpb` below 0.982763 by adapting variance normalization faster.

INTENDED_EDIT: Restore baseline weight decay for all Muon groups, retain the verified-best `sqrt(7/8)` contraction learning rate, and halve only contraction matrices’ variance-estimation horizon.

EVIDENCE: Projection-only `sqrt(7/8)` reached 0.982763, while doubling its variance-estimation horizon with `beta2=sqrt(0.95)` regressed to 0.983340; testing the opposite horizon direction is the clearest remaining projection-specific optimizer ablation.

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
            projection_beta2 = 0.90 if is_mlp_projection else 0.95
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr * projection_scale,
                momentum=0.95, ns_steps=5, beta2=projection_beta2,
                weight_decay=weight_decay,
                weight_decay_scale=1.0,
            ))
>>>>>>> REPLACE