MECHANISM: Projection-only Muon-rate refinement

HYPOTHESIS: Setting the MLP contraction rate to `(7/8)^(3/8)` will retain at least 445M training tokens and reduce `val_bpb` below 0.982763.

INTENDED_EDIT: Interpolate the 512×2560 MLP projection learning rate halfway in log space between the verified square-root and fourth-root reductions.

EVIDENCE: The square-root reduction achieved the best `val_bpb` of 0.982763, while the nearby fourth-root reduction reached 0.982852 and the stronger 7/8 reduction regressed to 0.983602, motivating a finer search near the best rate on its less-reduced side.

<<<<<<< SEARCH
            # Isolate the component favored by prior ablations: lower the rate
            # only for 512x2560 MLP contraction weights.
            mlp_lr_scale = math.sqrt(7 / 8) if shape[1] == 5 * shape[0] else 1.0
=======
            # Refine the projection-only optimum halfway in log space between
            # the verified square-root and fourth-root rate reductions.
            mlp_lr_scale = (7 / 8) ** (3 / 8) if shape[1] == 5 * shape[0] else 1.0
>>>>>>> REPLACE