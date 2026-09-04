MECHANISM: Axis-only broad translation augmentation

HYPOTHESIS: Removing diagonal training shifts while retaining cardinal shifts up to two pixels will exceed 9,166 correct predictions by aligning augmentation with the successful cardinal validation views without sacrificing broad translation invariance.

INTENDED_EDIT: Sample each training translation along exactly one randomly chosen axis, preserving the existing triangular ±2 displacement distribution, flips, optimizer, model, and runtime profile.

EVIDENCE: Adding diagonal validation views reduced correctness from 9,166 to 9,155. The timed-out curriculum also changed phase and shift magnitude; this patch isolates diagonal-support removal while retaining the baseline’s broad augmentation.

<<<<<<< SEARCH
    offset_y = torch.randint(
        0, 3, (images.shape[0],), device=images.device
    ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
    offset_x = torch.randint(
        0, 3, (images.shape[0],), device=images.device
    ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
=======
    sampled_offset = torch.randint(
        0, 3, (images.shape[0],), device=images.device
    ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
    translation_axis = torch.randint(
        0, 2, (images.shape[0],), device=images.device
    )
    centered_offset = torch.full_like(sampled_offset, 2)
    offset_y = torch.where(
        translation_axis == 0, sampled_offset, centered_offset
    )
    offset_x = torch.where(
        translation_axis == 1, sampled_offset, centered_offset
    )
>>>>>>> REPLACE