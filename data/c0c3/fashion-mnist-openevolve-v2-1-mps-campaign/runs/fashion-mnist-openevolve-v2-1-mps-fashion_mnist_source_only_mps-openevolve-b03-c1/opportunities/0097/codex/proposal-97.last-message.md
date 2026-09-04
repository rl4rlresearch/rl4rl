MECHANISM: Validation-aligned cardinal translation sampling

HYPOTHESIS: Sampling training translations from the same center and cardinal ±1 distribution used by the successful validation ensemble will exceed 9,166 correct predictions by eliminating unsupported diagonal and ±2 crops.

INTENDED_EDIT: Replace independent triangular x/y translations with center/cardinal ±1 translations sampled in the validation ensemble’s exact 3:2:2:2:2 proportions; retain horizontal flips and all other behavior.

EVIDENCE: Adding diagonal validation views reduced correctness from 9,166 to 9,155, while the current independent x/y augmentation frequently creates diagonal crops; this directly tests whether aligning training support with the established axis-only ensemble improves classification.

<<<<<<< SEARCH
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    batch_indices = torch.arange(images.shape[0], device=images.device)
    offset_y = torch.randint(
        0, 3, (images.shape[0],), device=images.device
    ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
    offset_x = torch.randint(
        0, 3, (images.shape[0],), device=images.device
    ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
    patches = padded.unfold(2, 28, 1).unfold(3, 28, 1)
    images = patches[batch_indices, :, offset_y, offset_x]
=======
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    batch_indices = torch.arange(images.shape[0], device=images.device)
    view_indices = torch.randint(
        0, 11, (images.shape[0],), device=images.device
    )
    offset_y_options = torch.tensor(
        [1, 1, 1, 0, 0, 2, 2, 1, 1, 1, 1],
        device=images.device,
    )
    offset_x_options = torch.tensor(
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 2],
        device=images.device,
    )
    offset_y = offset_y_options[view_indices]
    offset_x = offset_x_options[view_indices]
    patches = padded.unfold(2, 28, 1).unfold(3, 28, 1)
    images = patches[batch_indices, :, offset_y, offset_x]
>>>>>>> REPLACE