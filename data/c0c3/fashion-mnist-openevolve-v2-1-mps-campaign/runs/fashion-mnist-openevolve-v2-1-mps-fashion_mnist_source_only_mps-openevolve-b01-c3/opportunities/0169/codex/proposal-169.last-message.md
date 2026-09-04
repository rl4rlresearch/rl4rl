MECHANISM: Selectively stronger cross-offset consistency

HYPOTHESIS: Raising only cross-offset consistency from 5% to 7.5% while preserving 5% exact-reflection consistency will exceed 9,318 correct predictions by extending the observed benefit of stronger translation alignment.

INTENDED_EDIT: Apply per-example consistency weights of 0.075 to cross-offset pairs and 0.05 to exact-reflection pairs.

EVIDENCE: At the same 25% cross-offset frequency, increasing cross-offset consistency from 2.5% produced 9,312 correct to 5% producing 9,318, while exact-reflection consistency remained successful at 5%; a selective intermediate increase tests whether that positive dose response continues without weakening reflection training.

<<<<<<< SEARCH
    return classification_loss + 0.05 * consistency_loss.mean()
=======
    consistency_weights = torch.where(
        cross_offset_mask,
        consistency_loss.new_tensor(0.075),
        consistency_loss.new_tensor(0.05),
    )
    return classification_loss + (
        consistency_weights * consistency_loss
    ).mean()
>>>>>>> REPLACE