MECHANISM: Mild stochastic cutout regularization

HYPOTHESIS: Applying 6×6 cutout to 25% of training images will exceed 9,318 correct predictions by improving partial-occlusion robustness while remaining milder than the over-regularizing 0.05 label smoothing.

INTENDED_EDIT: Add training-only random 6×6 zero-filled erasure after the existing translation and horizontal-flip augmentation.

EVIDENCE: Label smoothing of 0.03 improved correctness from 9,311 to 9,318, while stronger 0.05 smoothing fell to 9,312; this motivates a mild, complementary input-space regularizer rather than stronger target smoothing or altered inference aggregation.

<<<<<<< SEARCH
    flip = torch.rand(batch, 1, 1, 1, device=images.device) < 0.5
    images = torch.where(flip, images.flip(-1), images)
    return images, labels
=======
    flip = torch.rand(batch, 1, 1, 1, device=images.device) < 0.5
    images = torch.where(flip, images.flip(-1), images)

    erase = torch.rand(batch, 1, 1, 1, device=images.device) < 0.25
    erase_y = torch.randint(0, 23, (batch, 1, 1, 1), device=images.device)
    erase_x = torch.randint(0, 23, (batch, 1, 1, 1), device=images.device)
    rows = torch.arange(28, device=images.device).view(1, 1, 28, 1)
    columns = torch.arange(28, device=images.device).view(1, 1, 1, 28)
    erase_mask = (
        erase
        & (rows >= erase_y)
        & (rows < erase_y + 6)
        & (columns >= erase_x)
        & (columns < erase_x + 6)
    )
    images = images.masked_fill(erase_mask, 0.0)
    return images, labels
>>>>>>> REPLACE