MECHANISM: Cosine-decayed label smoothing with a nonzero floor

HYPOTHESIS: Decaying label smoothing from 0.02 to 0.01 will exceed 9,257 correct predictions by gaining the late boundary sharpening seen with decay-to-zero while retaining enough regularization to avoid its three-prediction regression.

INTENDED_EDIT: Add a cosine smoothing schedule spanning 0.02 to 0.01 and apply it consistently to every individual, paired-view, and ensemble cross-entropy term.

EVIDENCE: Decaying smoothing from 0.02 to zero reduced cross-entropy but scored 9,254 correct versus the current 9,257, indicating that late sharpening helps calibration but removing smoothing entirely is slightly too aggressive.

<<<<<<< SEARCH
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    offsets = torch.randint(0, 5, (2,))
=======
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    smoothing = 0.01 + 0.005 * (1.0 + math.cos(math.pi * progress))
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
        label_smoothing=smoothing,
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
        label_smoothing=smoothing,
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
        label_smoothing=smoothing,
    )
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
        label_smoothing=smoothing,
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
        label_smoothing=smoothing,
    )
>>>>>>> REPLACE