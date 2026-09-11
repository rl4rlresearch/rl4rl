MECHANISM: Cosine target hardening

HYPOTHESIS: Decaying label smoothing from 0.02 to zero will exceed 9,247 correct predictions by retaining early regularization while sharpening class boundaries as ensemble supervision strengthens late in training.

INTENDED_EDIT: Apply a cosine-decayed label-smoothing value to every individual, pair, and ensemble cross-entropy term.

EVIDENCE: Added flip-consistency regularization reduced correctness from 9,247 to 9,241, while EMA reduced it to 9,225; reducing late-stage target regularization tests whether the established model instead benefits from sharper supervised convergence.

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
=======
    full_individual_loss = F.cross_entropy(
        torch.cat((logits_a, logits_b, flipped_a, flipped_b), dim=0),
        labels.repeat(4),
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    central_individual_loss = F.cross_entropy(
        torch.cat((central_logits, flipped_central), dim=0),
        labels.repeat(2),
        label_smoothing=0.02,
    )
=======
    central_individual_loss = F.cross_entropy(
        torch.cat((central_logits, flipped_central), dim=0),
        labels.repeat(2),
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        labels.repeat(2),
        label_smoothing=0.02,
    )
    central_pair_loss = F.cross_entropy(
=======
        labels.repeat(2),
        label_smoothing=label_smoothing,
    )
    central_pair_loss = F.cross_entropy(
>>>>>>> REPLACE

<<<<<<< SEARCH
    central_pair_loss = F.cross_entropy(
        0.5 * (central_logits + flipped_central),
        labels,
        label_smoothing=0.02,
    )
=======
    central_pair_loss = F.cross_entropy(
        0.5 * (central_logits + flipped_central),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        labels,
        label_smoothing=0.02,
    )
=======
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE