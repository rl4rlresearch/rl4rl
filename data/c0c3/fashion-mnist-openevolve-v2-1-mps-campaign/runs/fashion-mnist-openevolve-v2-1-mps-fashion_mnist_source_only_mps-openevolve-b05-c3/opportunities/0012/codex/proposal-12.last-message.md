MECHANISM: Extended inference-aligned augmentation phase

HYPOTHESIS: Extending center/cardinal one-pixel augmentation from the final quarter to the final third of the validated batch-64 training run will exceed 9,141 correct predictions by increasing the EMA model’s exposure to inference-matched views while retaining broad ±2 translation regularization early in training.

INTENDED_EDIT: Adopt the validated batch-64 optimizer, EMA, and equal-weight ten-view ensemble, then begin inference-aligned cardinal augmentation after two-thirds rather than three-quarters of training.

EVIDENCE: Restricting the final quarter to center/cardinal views improved the batch-64 result from 9,128 to 9,141 correct; varying the duration of that successful phase is the most direct next test.

<<<<<<< SEARCH
BATCH_SIZE = 128
GRAD_CLIP_NORM = 2.0
BASE_LR = 3.0e-3
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 2.0
BASE_LR = 3.0e-3
>>>>>>> REPLACE

<<<<<<< SEARCH
        probability_sum = 2.0 * F.softmax(logits, dim=1)
        probability_sum.add_(
            F.softmax(self._forward_once(images.flip(-1)), dim=1),
            alpha=2.0,
        )
=======
        probability_sum = F.softmax(logits, dim=1)
        probability_sum.add_(
            F.softmax(self._forward_once(images.flip(-1)), dim=1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 12.0).clamp_min(1e-8).log()
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log()
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
    if step * 3 < total_steps * 2:
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