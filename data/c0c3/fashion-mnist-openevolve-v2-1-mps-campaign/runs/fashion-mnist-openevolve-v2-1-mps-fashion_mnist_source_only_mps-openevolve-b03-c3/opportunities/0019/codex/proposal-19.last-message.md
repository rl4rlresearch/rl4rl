MECHANISM: Narrow progressive flip-orbit supervision

HYPOTHESIS: A cosine curriculum from 65% to 85% flip-ensemble weight will exceed 9,237 correct predictions by preserving beneficial progression while concentrating training closer to the empirically favored 75% regime.

INTENDED_EDIT: Replace the static 80% ensemble-loss mixture with a cosine schedule centered at 75% and bounded between 65% and 85%.

EVIDENCE: Narrowing the curriculum from 50%–100% (9,235 correct) to 60%–90% (9,237 correct) improved accuracy, while static 75% reached 9,233; this motivates testing a narrower schedule without eliminating progression.

<<<<<<< SEARCH
    del step, total_steps
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
=======
    progress = min(step / max(total_steps, 1), 1.0)
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    return 0.20 * individual_loss + 0.80 * ensemble_loss
=======
    ensemble_weight = 0.75 - 0.10 * math.cos(math.pi * progress)
    return (
        (1.0 - ensemble_weight) * individual_loss
        + ensemble_weight * ensemble_loss
    )
>>>>>>> REPLACE