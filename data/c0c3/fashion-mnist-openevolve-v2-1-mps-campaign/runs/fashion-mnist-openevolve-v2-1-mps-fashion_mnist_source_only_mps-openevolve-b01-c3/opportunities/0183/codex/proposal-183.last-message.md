MECHANISM: Disentangled translation and reflection pairing

HYPOTHESIS: Using 37.5% cross-offset pairs while making those pairs differ only by translation will exceed 9,325 correct predictions by preserving the validated cross-offset dose while avoiding compounded flip-plus-offset disagreement.

INTENDED_EDIT: Restore the best 37.5% cross-offset exposure and uniform 5% consistency, but use matching flip orientation for cross-offset pairs and opposite orientation for same-offset pairs.

EVIDENCE: Uniform 37.5% cross-offset training with 5% consistency achieved the best verified result of 9,325 correct; stronger translation exposure helped, while attempts to refine pair difficulty did not produce a completed result, motivating a compute-neutral isolation of translation consistency.

<<<<<<< SEARCH
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.25
=======
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.375
>>>>>>> REPLACE

<<<<<<< SEARCH
    partner_views = torch.where(
        flip_mask[:, None, None, None],
        partner_views,
        partner_views.flip(-1),
    )
=======
    partner_flip_mask = torch.where(
        cross_offset_mask,
        flip_mask,
        ~flip_mask,
    )
    partner_views = torch.where(
        partner_flip_mask[:, None, None, None],
        partner_views.flip(-1),
        partner_views,
    )
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