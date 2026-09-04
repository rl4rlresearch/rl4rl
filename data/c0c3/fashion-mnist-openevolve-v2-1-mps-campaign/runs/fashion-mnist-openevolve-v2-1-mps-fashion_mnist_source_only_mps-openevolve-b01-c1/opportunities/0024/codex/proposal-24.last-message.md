MECHANISM: Evaluation-distribution Monte Carlo supervision

HYPOTHESIS: Adding a randomly sampled central translation and its mirror with the evaluation ensemble’s exact 90% full-range/10% central weighting will exceed 9,209 correct predictions by aligning training with both components of the successful validation ensemble.

INTENDED_EDIT: Extend four-view antithetic supervision to six views, using four full-range translated/mirrored logits for 90% of each loss and two central translated/mirrored logits for 10%.

EVIDENCE: Four-view translation-and-flip ensemble supervision improved correctness from 9,124 to 9,209, while evaluation additionally gives 10% weight to a central-translation ensemble that the current training objective does not explicitly model.

<<<<<<< SEARCH
    offsets = torch.randint(0, 5, (2,))
    offset_y, offset_x = int(offsets[0]), int(offsets[1])
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    view_a = padded[
        :,
        :,
        offset_y : offset_y + 28,
        offset_x : offset_x + 28,
    ]
    view_b = padded[
        :,
        :,
        4 - offset_y : 4 - offset_y + 28,
        4 - offset_x : 4 - offset_x + 28,
    ]
    logits = model(
        torch.cat(
            (view_a, view_b, view_a.flip(-1), view_b.flip(-1)),
            dim=0,
        )
    )
    logits_a, logits_b, flipped_a, flipped_b = logits.chunk(4, dim=0)
    individual_loss = F.cross_entropy(
        logits,
        labels.repeat(4),
        label_smoothing=0.02,
    )
    ensemble_loss = F.cross_entropy(
        0.25 * (logits_a + logits_b + flipped_a + flipped_b),
        labels,
        label_smoothing=0.02,
    )
    return 0.5 * individual_loss + 0.5 * ensemble_loss
=======
    offsets = torch.randint(0, 5, (2,))
    offset_y, offset_x = int(offsets[0]), int(offsets[1])
    central_offsets = torch.randint(1, 4, (2,))
    central_y, central_x = int(central_offsets[0]), int(central_offsets[1])
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    view_a = padded[
        :,
        :,
        offset_y : offset_y + 28,
        offset_x : offset_x + 28,
    ]
    view_b = padded[
        :,
        :,
        4 - offset_y : 4 - offset_y + 28,
        4 - offset_x : 4 - offset_x + 28,
    ]
    central_view = padded[
        :,
        :,
        central_y : central_y + 28,
        central_x : central_x + 28,
    ]
    logits = model(
        torch.cat(
            (
                view_a,
                view_b,
                view_a.flip(-1),
                view_b.flip(-1),
                central_view,
                central_view.flip(-1),
            ),
            dim=0,
        )
    )
    (
        logits_a,
        logits_b,
        flipped_a,
        flipped_b,
        central_logits,
        flipped_central,
    ) = logits.chunk(6, dim=0)
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
    individual_loss = (
        0.9 * full_individual_loss + 0.1 * central_individual_loss
    )
    ensemble_loss = F.cross_entropy(
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        labels,
        label_smoothing=0.02,
    )
    return 0.5 * individual_loss + 0.5 * ensemble_loss
>>>>>>> REPLACE