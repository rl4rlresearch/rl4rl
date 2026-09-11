MECHANISM: Second-pass deterministic translation augmentation

HYPOTHESIS: Training on balanced one-pixel translations during the second exposure will exceed 9,281 correct predictions by improving spatial robustness without changing the successful architecture or first-pass optimization.

INTENDED_EDIT: Preserve the first exposure unchanged, apply balanced replicate-padded 3×3 crops during the second exposure, and restore the best-verified evaluation scale of 1.20514.

EVIDENCE: Channel-mixing, gating, and depthwise architectural changes all underperformed the 9,281-correct baseline, motivating an orthogonal input-invariance test while retaining its verified smoothing and calibration.

<<<<<<< SEARCH
        if not self.training:
            logits = 1.17817 * logits
=======
        if not self.training:
            logits = 1.20514 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    return images, labels
=======
) -> tuple[torch.Tensor, torch.Tensor]:
    if step >= total_steps // 2:
        height, width = images.shape[-2:]
        patches = F.unfold(
            F.pad(images, (1, 1, 1, 1), mode="replicate"),
            kernel_size=(height, width),
        ).view(
            images.shape[0],
            images.shape[1],
            height,
            width,
            9,
        )
        crop_indices = (
            torch.arange(images.shape[0], device=images.device) + step
        ) % 9
        gather_indices = crop_indices.view(-1, 1, 1, 1, 1).expand(
            -1,
            images.shape[1],
            height,
            width,
            1,
        )
        images = patches.gather(4, gather_indices).squeeze(4)
    return images, labels
>>>>>>> REPLACE