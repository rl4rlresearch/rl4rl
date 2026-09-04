MECHANISM: Three-eighths inference-aligned augmentation phase

HYPOTHESIS: Beginning center/cardinal one-pixel augmentation after five-eighths of training will exceed 9,155 correct predictions by extending the beneficial inference-matched phase beyond one-third while retaining more broad ±2 translation exposure than the timed-out two-fifths configuration.

INTENDED_EDIT: Preserve the validated batch-64 model, optimizer, EMA, loss, and ten-view ensemble while using broad ±2 translations for the first five-eighths of training and center/cardinal one-pixel translations for the final three-eighths.

EVIDENCE: Extending the cardinal phase from the final quarter to the final third increased validation correct from 9,141 to 9,155 and reduced cross-entropy from 0.24300 to 0.24135; three-eighths is the untested midpoint between that best completed duration and the timed-out two-fifths attempt.

<<<<<<< SEARCH
    del step, total_steps
    batch_indices = torch.arange(images.shape[0], device=images.device)
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    crops = padded.unfold(2, 28, 1).unfold(3, 28, 1)
    offsets_y = torch.randint(0, 5, (images.shape[0],), device=images.device)
    offsets_x = torch.randint(0, 5, (images.shape[0],), device=images.device)
    images = crops[batch_indices, :, offsets_y, offsets_x]
=======
    batch_indices = torch.arange(images.shape[0], device=images.device)
    if step * 8 < total_steps * 5:
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        crops = padded.unfold(2, 28, 1).unfold(3, 28, 1)
        offsets_y = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
        offsets_x = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
    else:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        crops = padded.unfold(2, 28, 1).unfold(3, 28, 1)
        directions = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
        offsets_y = (
            1 + (directions == 2).long() - (directions == 1).long()
        )
        offsets_x = (
            1 + (directions == 4).long() - (directions == 3).long()
        )
    images = crops[batch_indices, :, offsets_y, offsets_x]
>>>>>>> REPLACE