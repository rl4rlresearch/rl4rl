MECHANISM: Intermediate geometric–arithmetic view pooling

HYPOTHESIS: Increasing the validated probability-pooling blend from 12.5% to 18.75% will exceed 9,325 correct predictions or, if correctness ties, reduce cross-entropy below 0.193040839 by further damping unreliable extreme view logits.

INTENDED_EDIT: Restore the best uniform 37.5% cross-offset training with 5% consistency, then evaluate with an 81.25% logit-pooled and 18.75% arithmetic-probability ensemble.

EVIDENCE: Uniform 37.5% cross-offset training with a 12.5% probability blend achieved the best verified score, 9,325 correct at 0.193040839 cross-entropy; the 25% attempt timed out without negative accuracy evidence, motivating an intermediate fixed dose.

<<<<<<< SEARCH
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        pooled_predictions = pooled_logits.argmax(dim=-1)
=======
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        offset_probabilities = view_logits.softmax(dim=-1).mean(dim=1)
        pooled_probabilities = (
            offset_weights.unsqueeze(-1) * offset_probabilities
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        hybrid_probabilities = (
            0.8125 * pooled_logits.softmax(dim=-1)
            + 0.1875 * pooled_probabilities
        )
        pooled_logits = hybrid_probabilities.clamp_min(1.0e-8).log()
        pooled_predictions = pooled_logits.argmax(dim=-1)
>>>>>>> REPLACE

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
    return classification_loss + (
        consistency_weights * consistency_loss
    ).mean()
=======
    return classification_loss + 0.05 * consistency_loss.mean()
>>>>>>> REPLACE