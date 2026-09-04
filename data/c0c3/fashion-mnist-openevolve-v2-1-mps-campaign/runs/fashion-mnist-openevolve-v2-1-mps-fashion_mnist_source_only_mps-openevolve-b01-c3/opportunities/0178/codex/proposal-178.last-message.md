MECHANISM: Conservative joint orbit-regularization refinement

HYPOTHESIS: Combining the validated 37.5% cross-offset exposure with a modest 5.625% cross-offset consistency weight will exceed 9,325 correct predictions while preserving 5% exact-reflection regularization.

INTENDED_EDIT: Increase cross-offset pair frequency from 25% to 37.5% and selectively raise only their consistency weight from 2.5% to 5.625%.

EVIDENCE: At 25% exposure, raising cross-offset consistency from 2.5% to 5% improved correctness from 9,312 to 9,318; raising exposure to 37.5% at 5% further improved it to 9,325. This tests a conservative continuation of both validated dose responses.

<<<<<<< SEARCH
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.25
=======
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.375
>>>>>>> REPLACE

<<<<<<< SEARCH
    consistency_weights = torch.where(
        cross_offset_mask,
        consistency_loss.new_tensor(0.025),
        consistency_loss.new_tensor(0.05),
    )
=======
    consistency_weights = torch.where(
        cross_offset_mask,
        consistency_loss.new_tensor(0.05625),
        consistency_loss.new_tensor(0.05),
    )
>>>>>>> REPLACE