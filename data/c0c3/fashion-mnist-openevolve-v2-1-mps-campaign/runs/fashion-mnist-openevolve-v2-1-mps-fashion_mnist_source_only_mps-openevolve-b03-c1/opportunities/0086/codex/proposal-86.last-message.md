MECHANISM: Translation-range curriculum

HYPOTHESIS: Restricting random translations from ±2 pixels to ±1 during the final quarter will exceed 9,166 correct predictions by preserving early translation robustness while aligning late optimization with the validation-time views.

INTENDED_EDIT: Keep the existing triangular ±2 translation augmentation for the first 75% of training, then use uniformly sampled ±1 translations for the final 25%; retain horizontal flips and all model, loss, optimizer, and EMA behavior.

EVIDENCE: Weak mixup reduced correctness from 9,166 to 9,130 despite a clean loss tail, showing sensitivity to augmentation-induced distribution mismatch under the fixed exposure budget; this tests a targeted late-stage alignment of the existing spatial augmentation rather than adding regularization or model computation.

<<<<<<< SEARCH
    del step, total_steps
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    batch_indices = torch.arange(images.shape[0], device=images.device)
    offset_y = torch.randint(
        0, 3, (images.shape[0],), device=images.device
    ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
    offset_x = torch.randint(
        0, 3, (images.shape[0],), device=images.device
    ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
=======
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    batch_indices = torch.arange(images.shape[0], device=images.device)
    if 4 * (step + 1) < 3 * total_steps:
        offset_y = torch.randint(
            0, 3, (images.shape[0],), device=images.device
        ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
        offset_x = torch.randint(
            0, 3, (images.shape[0],), device=images.device
        ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
    else:
        offset_y = torch.randint(
            1, 4, (images.shape[0],), device=images.device
        )
        offset_x = torch.randint(
            1, 4, (images.shape[0],), device=images.device
        )
>>>>>>> REPLACE