MECHANISM: Moderate central-crop emphasis

HYPOTHESIS: Increasing central-crop allocation from 10% to 12.5% will exceed 9,247 correct predictions by modestly favoring less-displaced views while preserving the successful full-crop ensemble.

INTENDED_EDIT: Reweight validation aggregation and all corresponding training objectives from 90/10 to 87.5/12.5 full-versus-central allocation.

EVIDENCE: Deterministic crop balancing, crop microbatching, and added antithetic-crop supervision regressed to 9,238, 9,237, and 9,240 correct; refining the existing validation-aligned crop mixture tests a cheaper, narrower augmentation axis without disturbing the 9,247-correct curriculum.

<<<<<<< SEARCH
        return 1.29834 * (0.9 * full_ensemble + 0.1 * central_ensemble)
=======
        return 1.29834 * (
            0.875 * full_ensemble + 0.125 * central_ensemble
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    individual_loss = (
        0.9 * full_individual_loss + 0.1 * central_individual_loss
    )
=======
    individual_loss = (
        0.875 * full_individual_loss + 0.125 * central_individual_loss
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    pair_loss = 0.9 * full_pair_loss + 0.1 * central_pair_loss
=======
    pair_loss = 0.875 * full_pair_loss + 0.125 * central_pair_loss
>>>>>>> REPLACE

<<<<<<< SEARCH
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
=======
        0.21875 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.0625 * (central_logits + flipped_central),
>>>>>>> REPLACE