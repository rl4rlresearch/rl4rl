MECHANISM: Probabilistic mild rotation augmentation

HYPOTHESIS: Rotating half of the training images by up to 8 degrees will exceed 9,257 correct predictions by learning orientation robustness complementary to the existing translation and flip augmentation.

INTENDED_EDIT: Apply per-image bilinear rotation with reflection padding before generating the six translated and flipped training views; leave the architecture and inference ensemble unchanged.

EVIDENCE: Crop-weighted evaluation demonstrated sensitivity to geometric variation, while further widening, mixed pooling, classifier dropout, and smaller-batch optimization all regressed; mild rotation tests an unexplored geometric invariance without consuming parameters.

<<<<<<< SEARCH
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    offsets = torch.randint(0, 5, (2,))
=======
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    angles = torch.empty(
        images.shape[0],
        device=images.device,
        dtype=images.dtype,
    ).uniform_(-8.0, 8.0)
    angles = angles * (
        torch.rand(images.shape[0], device=images.device) < 0.5
    ).to(images.dtype)
    radians = angles * (math.pi / 180.0)
    cosines = radians.cos()
    sines = radians.sin()
    zeros = torch.zeros_like(cosines)
    theta = torch.stack(
        (cosines, -sines, zeros, sines, cosines, zeros),
        dim=1,
    ).reshape(-1, 2, 3)
    rotation_grid = F.affine_grid(
        theta,
        images.shape,
        align_corners=False,
    )
    images = F.grid_sample(
        images,
        rotation_grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )
    offsets = torch.randint(0, 5, (2,))
>>>>>>> REPLACE