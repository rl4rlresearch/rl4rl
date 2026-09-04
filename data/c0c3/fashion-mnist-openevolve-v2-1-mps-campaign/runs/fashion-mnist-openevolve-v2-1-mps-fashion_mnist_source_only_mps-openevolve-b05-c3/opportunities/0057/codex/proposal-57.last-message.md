MECHANISM: Dense-local augmentation bridge

HYPOTHESIS: Replacing the final 1/64 of broad ±2 augmentation with dense ±1 translations will exceed 9,172 correct predictions by easing distribution alignment without prematurely imposing the cardinal-only restriction.

INTENDED_EDIT: Use broad ±2 translations through 37/64, all nine ±1 translations from 37/64 to 38/64, then retain the verified center/cardinal terminal phase.

EVIDENCE: The 38/64 boundary achieved 9,172 correct, while moving cardinal-only augmentation directly to 37/64 fell to 9,160; a mild local bridge tests whether that interval benefits from reduced translation magnitude rather than full terminal restriction.

<<<<<<< SEARCH
    if step * 32 < total_steps * 19:
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        crops = padded.unfold(2, 28, 1).unfold(3, 28, 1)
        offsets_y = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
        offsets_x = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
    else:
=======
    if step * 64 < total_steps * 37:
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        crops = padded.unfold(2, 28, 1).unfold(3, 28, 1)
        offsets_y = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
        offsets_x = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
    elif step * 32 < total_steps * 19:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        crops = padded.unfold(2, 28, 1).unfold(3, 28, 1)
        offsets_y = torch.randint(
            0, 3, (images.shape[0],), device=images.device
        )
        offsets_x = torch.randint(
            0, 3, (images.shape[0],), device=images.device
        )
    else:
>>>>>>> REPLACE