MECHANISM: Reduced central-crop emphasis

HYPOTHESIS: Decreasing central-crop allocation from 10% to 7.5% will exceed 9,247 correct predictions by favoring the broader crop distribution after increasing central allocation to 12.5% regressed to 9,239.

INTENDED_EDIT: Reweight validation aggregation and all corresponding training objectives from 90/10 to 92.5/7.5 full-versus-central allocation.

EVIDENCE: Raising central-crop allocation from 10% to 12.5% reduced validation correctness by eight images, providing directional evidence that the successful ensemble may benefit from less central-crop emphasis.

<<<<<<< SEARCH
        return 1.29834 * (0.9 * full_ensemble + 0.1 * central_ensemble)
=======
        return 1.29834 * (0.925 * full_ensemble + 0.075 * central_ensemble)
>>>>>>> REPLACE

<<<<<<< SEARCH
    individual_loss = (
        0.9 * full_individual_loss + 0.1 * central_individual_loss
    )
=======
    individual_loss = (
        0.925 * full_individual_loss + 0.075 * central_individual_loss
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    pair_loss = 0.9 * full_pair_loss + 0.1 * central_pair_loss
=======
    pair_loss = 0.925 * full_pair_loss + 0.075 * central_pair_loss
>>>>>>> REPLACE

<<<<<<< SEARCH
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
=======
        0.23125 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.0375 * (central_logits + flipped_central),
>>>>>>> REPLACE