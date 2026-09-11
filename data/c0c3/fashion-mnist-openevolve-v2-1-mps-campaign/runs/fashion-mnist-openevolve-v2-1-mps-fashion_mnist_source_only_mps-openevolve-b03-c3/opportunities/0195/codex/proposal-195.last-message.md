MECHANISM: Mild rotational augmentation

HYPOTHESIS: Adding independent ±8° rotations after the existing translations will increase validation_correct above 9,284 by learning small orientation robustness not covered by prior augmentation, architecture, or ensemble experiments.

INTENDED_EDIT: Apply bilinear, border-padded random rotations during training and use the best verified evaluation scale of 1.16727.

EVIDENCE: Translation-distribution alignment fell to 9,262 correct and several architectural refinements regressed, while calibration has saturated at 9,284; this motivates testing a distinct label-preserving augmentation while retaining the validated model and TTA.

<<<<<<< SEARCH
        return 1.1682 * logits
=======
        return 1.16727 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    images = images.gather(3, column_index)

    return images, labels
=======
    images = images.gather(3, column_index)

    max_angle = math.radians(8.0)
    angles = images.new_empty(images.size(0)).uniform_(
        -max_angle, max_angle
    )
    cosine = angles.cos()
    sine = angles.sin()
    transform = images.new_zeros((images.size(0), 2, 3))
    transform[:, 0, 0] = cosine
    transform[:, 0, 1] = -sine
    transform[:, 1, 0] = sine
    transform[:, 1, 1] = cosine
    grid = F.affine_grid(transform, images.shape, align_corners=False)
    images = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )

    return images, labels
>>>>>>> REPLACE