MECHANISM: Antithetic central-translation ensemble supervision

HYPOTHESIS: Supervising complementary central translations and their mirrors will exceed 9,218 correct predictions by reducing Monte Carlo mismatch with the validation ensemble’s nine-offset central component.

INTENDED_EDIT: Expand the central training sample from one translated flip pair to two complementary translated flip pairs, preserving the evaluation-aligned 90% full-range and 10% central weighting.

EVIDENCE: Adding one randomly sampled central translation and its mirror improved correctness from 9,209 to 9,218; the full-range component already benefits from complementary translations, while the central component does not.

<<<<<<< SEARCH
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
=======
    central_view_a = padded[
        :,
        :,
        central_y : central_y + 28,
        central_x : central_x + 28,
    ]
    central_view_b = padded[
        :,
        :,
        4 - central_y : 4 - central_y + 28,
        4 - central_x : 4 - central_x + 28,
    ]
    logits = model(
        torch.cat(
            (
                view_a,
                view_b,
                view_a.flip(-1),
                view_b.flip(-1),
                central_view_a,
                central_view_b,
                central_view_a.flip(-1),
                central_view_b.flip(-1),
            ),
            dim=0,
        )
    )
    (
        logits_a,
        logits_b,
        flipped_a,
        flipped_b,
        central_logits_a,
        central_logits_b,
        flipped_central_a,
        flipped_central_b,
    ) = logits.chunk(8, dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    central_individual_loss = F.cross_entropy(
        torch.cat((central_logits, flipped_central), dim=0),
        labels.repeat(2),
        label_smoothing=0.02,
    )
=======
    central_individual_loss = F.cross_entropy(
        torch.cat(
            (
                central_logits_a,
                central_logits_b,
                flipped_central_a,
                flipped_central_b,
            ),
            dim=0,
        ),
        labels.repeat(4),
        label_smoothing=0.02,
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
        + 0.025
        * (
            central_logits_a
            + central_logits_b
            + flipped_central_a
            + flipped_central_b
        ),
        labels,
        label_smoothing=0.02,
    )
>>>>>>> REPLACE