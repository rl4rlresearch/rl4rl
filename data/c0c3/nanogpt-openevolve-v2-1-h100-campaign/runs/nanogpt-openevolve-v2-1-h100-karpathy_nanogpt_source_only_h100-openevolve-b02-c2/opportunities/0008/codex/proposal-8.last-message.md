MECHANISM: Token-horizon NorMuon variance averaging

HYPOTHESIS: Restoring the proven 300-step momentum ramp while increasing NorMuon beta2 from 0.95 to 0.975 will lower val_bpb below 0.987182 by smoothing noisy matrix-update statistics over approximately the same token horizon as the original 524K-token batch.

INTENDED_EDIT: Restore the best-performing Muon momentum ramp and lengthen only NorMuon’s second-moment averaging for the 262K-token batch.

EVIDENCE: The 262K-token design with a 300-step ramp achieved the best val_bpb of 0.987182, while extending that ramp to 600 steps worsened val_bpb to 0.988827. Halving the batch doubles updates per token, motivating a separate test of longer variance-statistic averaging without altering the successful first-moment trajectory.

<<<<<<< SEARCH
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
=======
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.975, weight_decay=weight_decay,
            ))
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_muon_momentum(step):
    # Preserve the 524K-batch momentum trajectory in tokens after halving the batch.
    frac = min(step / 600, 1)
    return (1 - frac) * 0.85 + frac * 0.95
=======
def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95
>>>>>>> REPLACE