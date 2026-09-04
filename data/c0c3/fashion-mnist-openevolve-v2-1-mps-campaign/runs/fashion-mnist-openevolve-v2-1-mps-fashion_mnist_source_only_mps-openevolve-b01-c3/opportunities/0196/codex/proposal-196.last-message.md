MECHANISM: Continued flip-disagreement reliability gating

HYPOTHESIS: Increasing the flip-inconsistency penalty from 12.5% to 15% will preserve 9,325 correct predictions while reducing validation cross-entropy below 0.1929871063.

INTENDED_EDIT: Restore uniform 37.5% cross-offset training and hybrid probability pooling, retain the verified 1.24754 calibration, and increase only the best design’s flip-disagreement penalty to 15%.

EVIDENCE: Moving from 10% gating at 0.1929895668 cross-entropy to 12.5% gating at 0.1929871063 improved the objective without changing the 9,325 correct predictions, motivating a conservative continuation in the same direction.

<<<<<<< SEARCH
        offset_weights = base_weights * (0.90 + 0.10 * flip_consistency)
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        pooled_predictions = pooled_logits.argmax(dim=-1)
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
        agreement_uncertainty = view_agreement * (1.0 - view_agreement)
        calibration = 1.22775 * (
            0.92211476
            + 0.07788524 * view_agreement
            - 0.02 * agreement_uncertainty
        )
        return calibration.unsqueeze(1) * pooled_logits
=======
        offset_weights = base_weights * (0.85 + 0.15 * flip_consistency)
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        offset_probabilities = view_logits.softmax(dim=-1).mean(dim=1)
        pooled_probabilities = (
            offset_weights.unsqueeze(-1) * offset_probabilities
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        hybrid_probabilities = (
            0.875 * pooled_logits.softmax(dim=-1)
            + 0.125 * pooled_probabilities
        )
        pooled_logits = hybrid_probabilities.clamp_min(1.0e-8).log()
        pooled_predictions = pooled_logits.argmax(dim=-1)
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
        agreement_uncertainty = view_agreement * (1.0 - view_agreement)
        calibration = 1.24754 * (
            0.92211476
            + 0.07788524 * view_agreement
            - 0.02 * agreement_uncertainty
        )
        return calibration.unsqueeze(1) * pooled_logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    positions = torch.randint(0, 5, (batch,), device=images.device)
    random_alternate_positions = (
        positions
        + torch.randint(1, 5, (batch,), device=images.device)
    ) % 5
    center_alternate_positions = torch.where(
        positions == 0,
        torch.randint(1, 5, (batch,), device=images.device),
        torch.zeros_like(positions),
    )
    center_anchor_mask = torch.rand(batch, device=images.device) < 0.7137
    alternate_positions = torch.where(
        center_anchor_mask,
        center_alternate_positions,
        random_alternate_positions,
    )
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.375
=======
    positions = torch.randint(0, 5, (batch,), device=images.device)
    alternate_positions = (
        positions
        + torch.randint(1, 5, (batch,), device=images.device)
    ) % 5
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.375
>>>>>>> REPLACE