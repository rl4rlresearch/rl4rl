MECHANISM: Two-stage augmentation-support curriculum

HYPOTHESIS: Using broad ±2 translations during the first half of training, then matching the successful center/cardinal validation-view distribution during the second half, will exceed 9,166 correct predictions by retaining early invariance while removing unsupported diagonal and large-shift exposure during late fitting.

INTENDED_EDIT: Preserve the existing augmentation initially, then switch halfway through training to center and cardinal ±1 crops sampled in the validation ensemble’s 3:2:2:2:2 proportions.

EVIDENCE: Adding diagonal validation views reduced correctness from 9,166 to 9,155, indicating that diagonal shifts dilute useful predictions; a late alignment stage tests that signal without discarding the baseline’s broad early regularization.

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
    if 2 * step < total_steps:
        offset_y = torch.randint(
            0, 3, (images.shape[0],), device=images.device
        ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
        offset_x = torch.randint(
            0, 3, (images.shape[0],), device=images.device
        ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
    else:
        view_choice = torch.randint(
            0, 11, (images.shape[0],), device=images.device
        )
        offset_y = torch.full_like(view_choice, 2)
        offset_x = torch.full_like(view_choice, 2)
        offset_y = torch.where(
            (view_choice >= 3) & (view_choice < 5), 1, offset_y
        )
        offset_y = torch.where(
            (view_choice >= 5) & (view_choice < 7), 3, offset_y
        )
        offset_x = torch.where(
            (view_choice >= 7) & (view_choice < 9), 1, offset_x
        )
        offset_x = torch.where(view_choice >= 9, 3, offset_x)
>>>>>>> REPLACE