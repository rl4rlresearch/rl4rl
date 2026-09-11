MECHANISM: Cosine target hardening

HYPOTHESIS: Decaying label smoothing from 0.02 to zero will exceed 9,247 correct predictions by preserving early regularization while sharpening class boundaries as ensemble supervision strengthens late in training.

INTENDED_EDIT: Compute a cosine-decayed smoothing coefficient and apply it consistently to all individual, pair, and ensemble cross-entropy terms.

EVIDENCE: The prior target-hardening implementation could not be verified, leaving its hypothesis untested; meanwhile added consistency and EMA regularization regressed to 9,241 and 9,225 correct, respectively, motivating reduced late regularization in the established 9,247-correct design.

<<<<<<< SEARCH
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    offsets = torch.randint(0, 5, (2,))
=======
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    label_smoothing = 0.01 * (1.0 + math.cos(math.pi * progress))
    offsets = torch.randint(0, 5, (2,))
>>>>>>> REPLACE

<<<<<<< SEARCH
    full_individual_loss = F.cross_entropy(
        torch.cat((logits_a, logits_b, flipped_a, flipped_b), dim=0),
        labels.repeat(4),
        label_smoothing=0.02,
    )
    central_individual_loss = F.cross_entropy(
        torch.cat((central_logits, flipped_central), dim=0),
        labels.repeat(2),
        label_smoothing=0.02,
    )
=======
    full_individual_loss = F.cross_entropy(
        torch.cat((logits_a, logits_b, flipped_a, flipped_b), dim=0),
        labels.repeat(4),
        label_smoothing=label_smoothing,
    )
    central_individual_loss = F.cross_entropy(
        torch.cat((central_logits, flipped_central), dim=0),
        labels.repeat(2),
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    full_pair_loss = F.cross_entropy(
        torch.cat(
            (
                0.5 * (logits_a + flipped_a),
                0.5 * (logits_b + flipped_b),
            ),
            dim=0,
        ),
        labels.repeat(2),
        label_smoothing=0.02,
    )
    central_pair_loss = F.cross_entropy(
        0.5 * (central_logits + flipped_central),
        labels,
        label_smoothing=0.02,
    )
=======
    full_pair_loss = F.cross_entropy(
        torch.cat(
            (
                0.5 * (logits_a + flipped_a),
                0.5 * (logits_b + flipped_b),
            ),
            dim=0,
        ),
        labels.repeat(2),
        label_smoothing=label_smoothing,
    )
    central_pair_loss = F.cross_entropy(
        0.5 * (central_logits + flipped_central),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    ensemble_loss = F.cross_entropy(
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        labels,
        label_smoothing=0.02,
    )
=======
    ensemble_loss = F.cross_entropy(
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE