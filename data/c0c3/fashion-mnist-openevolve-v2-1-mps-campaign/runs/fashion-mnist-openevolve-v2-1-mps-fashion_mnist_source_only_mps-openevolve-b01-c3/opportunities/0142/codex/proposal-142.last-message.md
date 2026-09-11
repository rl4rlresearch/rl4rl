MECHANISM: Flip-consistency-weighted logit ensembling

HYPOTHESIS: Downweighting only translation offsets whose original and flipped views disagree will exceed 9,287 correct predictions by suppressing locally unreliable evidence while reproducing the verified pooling exactly for flip-stable offsets.

INTENDED_EDIT: Replace fixed offset pooling with a mild 15% penalty for flip-inconsistent view pairs, while retaining the verified center-view preference and agreement-conditioned calibration.

EVIDENCE: Agreement-conditioned calibration showed that view consensus contains useful reliability information, while wholesale probability pooling lost one correct prediction; using consensus only to adjust unreliable offset pairs is a narrower aggregation change.

<<<<<<< SEARCH
        offset_logits = view_logits.mean(dim=1)
        pooled_logits = (
            1.5578022218542173 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578022218542173
        pooled_predictions = pooled_logits.argmax(dim=-1)
=======
        offset_logits = view_logits.mean(dim=1)
        flip_agreement = (
            view_logits[:, 0].argmax(dim=-1)
            == view_logits[:, 1].argmax(dim=-1)
        ).float()
        offset_weights = 0.85 + 0.15 * flip_agreement
        offset_weights[0] = 1.5578022218542173 * offset_weights[0]
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0, keepdim=True)
        pooled_predictions = pooled_logits.argmax(dim=-1)
>>>>>>> REPLACE