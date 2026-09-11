MECHANISM: Flip-consistency-weighted logit ensembling

HYPOTHESIS: A 15% penalty on flip-inconsistent offset pairs will exceed 9,287 correct predictions by suppressing locally unreliable transformed evidence while leaving flip-stable pooling unchanged.

INTENDED_EDIT: Replace fixed offset averaging with per-image offset weights derived from agreement between each offset’s original and flipped predictions, retaining the verified center preference and calibration.

EVIDENCE: Agreement-conditioned calibration improved cross-entropy without changing predictions, showing view consensus carries reliability information; probability pooling lost one correct prediction, motivating this narrower unresolved aggregation change.

<<<<<<< SEARCH
        offset_logits = view_logits.mean(dim=1)
        pooled_logits = (
            1.5578022218542173 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578022218542173
        pooled_predictions = pooled_logits.argmax(dim=-1)
=======
        offset_logits = view_logits.mean(dim=1)
        flip_consistency = (
            view_logits[:, 0].argmax(dim=-1)
            == view_logits[:, 1].argmax(dim=-1)
        ).float()
        base_weights = offset_logits.new_tensor(
            (1.5578022218542173, 1.0, 1.0, 1.0, 1.0)
        ).unsqueeze(1)
        offset_weights = base_weights * (0.85 + 0.15 * flip_consistency)
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        pooled_predictions = pooled_logits.argmax(dim=-1)
>>>>>>> REPLACE