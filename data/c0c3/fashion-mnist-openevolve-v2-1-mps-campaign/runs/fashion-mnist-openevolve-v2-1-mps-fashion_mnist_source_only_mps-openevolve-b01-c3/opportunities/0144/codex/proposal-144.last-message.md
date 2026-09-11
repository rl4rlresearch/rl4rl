MECHANISM: Flip-consistency-weighted logit ensembling

HYPOTHESIS: Applying the verified 15% penalty to flip-inconsistent offset pairs will increase validation correctness from 9,287 to 9,289 while retaining agreement-conditioned calibration.

INTENDED_EDIT: Replace fixed offset averaging and global scaling with the best verified flip-consistency weighting, center weight, and per-image agreement calibration.

EVIDENCE: Reference Design 2 achieved the highest available result—9,289 correct—whereas fixed pooling produced 9,287; probability pooling and learned feature recalibration were worse.

<<<<<<< SEARCH
        offset_logits = view_logits.mean(dim=1)
        pooled_logits = (
            1.557939852073332 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.557939852073332
        return 1.22775 * pooled_logits
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
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
        calibration = 1.22775 * (
            0.92211476 + 0.07788524 * view_agreement
        )
        return calibration.unsqueeze(1) * pooled_logits
>>>>>>> REPLACE