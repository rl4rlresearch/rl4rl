MECHANISM: Low-overhead deterministic translation augmentation

HYPOTHESIS: Cycling through centered and four one-pixel translations will exceed 9,330 correct predictions by improving shift invariance without adding parameters or model-forward work.

INTENDED_EDIT: Apply centered, up, down, left, and right training views using a single contiguous `torch.roll` operation per augmented batch.

EVIDENCE: The verified flip-invariant procedure achieved 9,330 correct; the prior translation experiment timed out without producing negative accuracy evidence, motivating a lower-overhead implementation of the same spatial-invariance idea.

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del total_steps
    shifts = (
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    )
    shift = shifts[step % len(shifts)]
    if shift != (0, 0):
        images = torch.roll(images, shifts=shift, dims=(-2, -1))
    return images, labels
>>>>>>> REPLACE