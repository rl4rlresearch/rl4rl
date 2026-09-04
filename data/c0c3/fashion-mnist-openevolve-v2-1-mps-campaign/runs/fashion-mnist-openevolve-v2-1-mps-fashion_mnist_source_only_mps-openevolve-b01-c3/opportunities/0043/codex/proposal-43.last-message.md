MECHANISM: Stratified transformation sampling with linear tail averaging

HYPOTHESIS: Exactly balancing the ten position/flip transformations within each minibatch will exceed 9,285 correct predictions by reducing augmentation-gradient variance, while the verified final-10% linear average preserves the strongest optimization baseline.

INTENDED_EDIT: Restore final-10% linearly recency-weighted parameter averaging and replace independent random augmentation draws with a deterministic, step-rotated balance across all ten training transformations.

EVIDENCE: Reference Design 2 achieved the best verified result of 9,285 correct using final-10% linear averaging and uniform transformation sampling; the current final-5% average fell to 9,275, while stratification makes each limited-exposure minibatch more closely match the uniform ten-view inference distribution without added model computation.

<<<<<<< SEARCH
    optimizer.tail_average_start = max(1, int(0.95 * total_steps))
=======
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    batch, channels, height, width = images.shape
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")

    positions = torch.randint(0, 5, (batch,), device=images.device)
=======
    del total_steps
    batch, channels, height, width = images.shape
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")

    transform_ids = (
        torch.arange(batch, device=images.device) + step * batch
    ).remainder(10)
    positions = transform_ids.remainder(5)
>>>>>>> REPLACE

<<<<<<< SEARCH
    flip_mask = torch.rand(batch, device=images.device) < 0.5
=======
    flip_mask = transform_ids >= 5
>>>>>>> REPLACE

<<<<<<< SEARCH
        average_weight = 1.0 / optimizer.tail_average_count
=======
        average_weight = 2.0 / (optimizer.tail_average_count + 1.0)
>>>>>>> REPLACE