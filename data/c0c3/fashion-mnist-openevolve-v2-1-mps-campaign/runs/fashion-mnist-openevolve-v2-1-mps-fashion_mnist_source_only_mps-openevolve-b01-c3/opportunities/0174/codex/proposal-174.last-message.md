MECHANISM: Intermediate-strength flip-offset orbit exposure

HYPOTHESIS: Increasing cross-offset pair frequency from the validated 37.5% to 43.75% while retaining uniform 5% consistency will exceed 9,325 correct predictions without changing model compute.

INTENDED_EDIT: Use cross-offset reflected partners for 43.75% of examples and apply the validated 5% consistency weight to every pair.

EVIDENCE: Raising cross-offset frequency from 25% to 37.5% at 5% consistency improved correctness from 9,318 to 9,325; 43.75% is a conservative midpoint toward the 50% attempt that timed out despite unchanged tensor shapes.

<<<<<<< SEARCH
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.25
=======
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.4375
>>>>>>> REPLACE

<<<<<<< SEARCH
    consistency_weights = torch.where(
        cross_offset_mask,
        consistency_loss.new_tensor(0.025),
        consistency_loss.new_tensor(0.05),
    )
    return classification_loss + (
        consistency_weights * consistency_loss
    ).mean()
=======
    return classification_loss + 0.05 * consistency_loss.mean()
>>>>>>> REPLACE