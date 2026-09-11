MECHANISM: Projection-only MLP Muon rate compensation

HYPOTHESIS: Reducing only the 512×2560 MLP projection learning rate by sqrt(4.375/5) will retain at least 445M training tokens and reduce val_bpb below 0.982780.

INTENDED_EDIT: Apply aspect compensation exclusively to MLP contraction matrices while leaving expansion and attention rates unchanged.

EVIDENCE: Symmetric MLP compensation improved val_bpb from 0.982905 to 0.982780, whereas expansion-only compensation regressed to 0.983257, indicating the reduced projection rate may be the beneficial component.

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
            # Isolate the component favored by prior ablations: lower the rate
            # only for 512x2560 MLP contraction weights.
            mlp_lr_scale = math.sqrt(7 / 8) if shape[1] == 5 * shape[0] else 1.0
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr * mlp_lr_scale,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
>>>>>>> REPLACE