MECHANISM: Per-example one-pixel translation augmentation

HYPOTHESIS: Replacing the redundant preparatory flip with random one-pixel translations will exceed 9,322 correct predictions by adding positional diversity while preserving exact paired horizontal-view supervision and the best verified top-four attention architecture.

INTENDED_EDIT: Generate an independently translated crop for each training image using replicate padding; the existing loss still trains on both that crop and its horizontal flip.

EVIDENCE: Exact top-four saliency achieved the best verified count of 9,322, while subsequent attention and checkpoint variants regressed or timed out; this preserves that architecture and tests a low-overhead, orthogonal augmentation improvement.

<<<<<<< SEARCH
    del step, total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )
    return images, labels
=======
    del step, total_steps
    height, width = images.shape[-2:]
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    crops = padded.unfold(2, height, 1).unfold(3, width, 1)
    crops = crops.permute(0, 2, 3, 1, 4, 5)
    offsets = torch.randint(
        0, 3, (images.shape[0], 2), device=images.device
    )
    batch_indices = torch.arange(images.shape[0], device=images.device)
    images = crops[batch_indices, offsets[:, 0], offsets[:, 1]]
    return images, labels
>>>>>>> REPLACE