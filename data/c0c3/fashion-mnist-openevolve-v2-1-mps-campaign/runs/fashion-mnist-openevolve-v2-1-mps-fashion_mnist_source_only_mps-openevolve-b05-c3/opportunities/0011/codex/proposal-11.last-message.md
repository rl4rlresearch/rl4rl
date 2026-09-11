MECHANISM: Late-phase inference-aligned cardinal augmentation

HYPOTHESIS: Using the validated batch-64 configuration while restricting the final quarter of training to the five center/cardinal one-pixel views used by inference will exceed 9,128 correct predictions by preserving early translation regularization and eliminating late optimization on the diagonal and two-pixel views that are mismatched with the best ensemble.

INTENDED_EDIT: Switch to batch size 64 with exposure-matched AdamW decay and EMA momentum, and replace uniform ±2 translation augmentation during the final training quarter with uniform center/cardinal one-pixel translations while retaining horizontal flips.

EVIDENCE: Batch size 64 achieved the best result at 9,128 correct, while adding diagonal inference views reduced correct predictions from 9,125 to 9,119; this motivates retaining the proven optimizer configuration and aligning EMA-dominant late training with the successful center/cardinal ensemble.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        weight_decay=3e-4,
=======
        weight_decay=1.5e-4,
>>>>>>> REPLACE

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
    if step * 4 < total_steps * 3:
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

<<<<<<< SEARCH
                average.lerp_(tensor, 0.01)
=======
                average.lerp_(tensor, 0.005)
>>>>>>> REPLACE