MECHANISM: Disagreement-adaptive geometric–arithmetic view pooling

HYPOTHESIS: Restoring uniform 37.5% cross-offset training and increasing the validated 12.5% probability blend only when flip views disagree will exceed 9,325 correct predictions by damping unreliable extreme logits without perturbing fully consistent examples.

INTENDED_EDIT: Remove detrimental center-anchored pair sampling, restore uniform cross-offset partners, and adapt the arithmetic-probability blend from 12.5% to at most 25% according to reliability-weighted flip disagreement.

EVIDENCE: Uniform 37.5% sampling with a 12.5% global probability blend achieved the best verified result of 9,325 correct and 0.1930408 cross-entropy, whereas center anchoring fell to 9,324; the 25% blend timed out rather than providing negative accuracy evidence, motivating a targeted interpolation only on uncertain examples.

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
        weighted_flip_agreement = (
            base_weights * flip_consistency
        ).sum(dim=0) / base_weights.sum()
        probability_blend = (
            0.125 + 0.125 * (1.0 - weighted_flip_agreement)
        )
        hybrid_probabilities = (
            (1.0 - probability_blend.unsqueeze(1))
            * pooled_logits.softmax(dim=-1)
            + probability_blend.unsqueeze(1) * pooled_probabilities
        )
        pooled_logits = hybrid_probabilities.clamp_min(1.0e-8).log()
        pooled_predictions = pooled_logits.argmax(dim=-1)
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