MECHANISM: Consensus-gated flip-pair suppression

HYPOTHESIS: Retaining the successful 10% penalty but increasing it to 25% only for flip-inconsistent offset pairs whose averaged prediction opposes the provisional ensemble will exceed 9,289 correct predictions.

INTENDED_EDIT: Use center-weighted provisional consensus to distinguish supportive from dissenting flip-inconsistent pairs, applying a 10% penalty to the former and 25% to the latter.

EVIDENCE: A uniform 10% pair-wide penalty achieved 9,289 correct, while uniform 25% retained that count but worsened cross-entropy and penalizing individual dissenting views fell to 9,287; pair-level semantic gating tests stronger suppression only where both inconsistency and ensemble opposition indicate unreliable evidence.

<<<<<<< SEARCH
        flip_consistency = (
            view_logits[:, 0].argmax(dim=-1)
            == view_logits[:, 1].argmax(dim=-1)
        ).float()
        base_weights = offset_logits.new_tensor(
            (1.5578022218542173, 1.0, 1.0, 1.0, 1.0)
        ).unsqueeze(1)
        offset_weights = base_weights * (0.90 + 0.10 * flip_consistency)
=======
        flip_consistency = (
            view_logits[:, 0].argmax(dim=-1)
            == view_logits[:, 1].argmax(dim=-1)
        ).float()
        base_weights = offset_logits.new_tensor(
            (1.5578022218542173, 1.0, 1.0, 1.0, 1.0)
        ).unsqueeze(1)
        provisional_logits = (
            base_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / base_weights.sum()
        provisional_predictions = provisional_logits.argmax(dim=-1)
        pair_dissent = (
            offset_logits.argmax(dim=-1)
            != provisional_predictions.unsqueeze(0)
        ).float()
        inconsistency_penalty = (
            0.10 + 0.15 * pair_dissent
        ) * (1.0 - flip_consistency)
        offset_weights = base_weights * (1.0 - inconsistency_penalty)
>>>>>>> REPLACE