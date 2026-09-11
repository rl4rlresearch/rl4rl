MECHANISM: Train–test geometric distribution alignment

HYPOTHESIS: Sampling centered images with probability 1/3 and each one-pixel cardinal translation with probability 1/6 will exceed 9,148 correct predictions by matching training augmentation to the empirically strongest center-weighted validation ensemble.

INTENDED_EDIT: Replace uniform ±2-pixel crop augmentation with a center-weighted mixture of centered and one-pixel cardinal views; retain random horizontal flips and all other settings.

EVIDENCE: The center-weighted cardinal ensemble reached 9,148 correct, outperforming equal weighting at 9,142 and the full translation grid at 9,137, indicating that centered and cardinal views are useful while diagonal or excessive translation influence is harmful.

<<<<<<< SEARCH
    batch = images.shape[0]
    offsets = torch.randint(0, 5, (batch, 2), device=images.device)
    windows = F.pad(images, (2, 2, 2, 2)).unfold(2, 28, 1).unfold(3, 28, 1)
    indices = torch.arange(batch, device=images.device)
    images = windows[indices, :, offsets[:, 0], offsets[:, 1], :, :]
=======
    batch = images.shape[0]
    choices = torch.randint(0, 6, (batch,), device=images.device)
    offset_y = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_x = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_y = torch.where(choices == 2, 0, offset_y)
    offset_y = torch.where(choices == 3, 2, offset_y)
    offset_x = torch.where(choices == 4, 0, offset_x)
    offset_x = torch.where(choices == 5, 2, offset_x)
    windows = F.pad(images, (1, 1, 1, 1)).unfold(2, 28, 1).unfold(3, 28, 1)
    indices = torch.arange(batch, device=images.device)
    images = windows[indices, :, offset_y, offset_x, :, :]
>>>>>>> REPLACE