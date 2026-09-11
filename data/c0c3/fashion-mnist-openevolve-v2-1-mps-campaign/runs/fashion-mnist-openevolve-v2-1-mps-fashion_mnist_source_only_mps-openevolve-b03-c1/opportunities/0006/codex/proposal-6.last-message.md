MECHANISM: Per-example translation diversification

HYPOTHESIS: Sampling an independent ±2-pixel translation for every training image will exceed 9,073 correct predictions by reducing correlated augmentation noise within each batch.

INTENDED_EDIT: Replace the single batch-wide crop offset with vectorized per-image crop selection while preserving the augmentation range, batch size, architecture, and optimizer.

EVIDENCE: Ten-view translation-and-flip inference improved the current design from 9,050 to 9,073 correct, showing translation variance remains consequential; training currently gives all 64 images the same translation, so independent offsets directly target that variance.

<<<<<<< SEARCH
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    offset_y = int(torch.randint(0, 5, (), device=images.device).item())
    offset_x = int(torch.randint(0, 5, (), device=images.device).item())
    images = padded[:, :, offset_y : offset_y + 28, offset_x : offset_x + 28]
=======
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    batch_indices = torch.arange(images.shape[0], device=images.device)
    offset_y = torch.randint(0, 5, (images.shape[0],), device=images.device)
    offset_x = torch.randint(0, 5, (images.shape[0],), device=images.device)
    patches = padded.unfold(2, 28, 1).unfold(3, 28, 1)
    images = patches[batch_indices, :, offset_y, offset_x]
>>>>>>> REPLACE