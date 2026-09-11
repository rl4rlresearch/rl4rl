MECHANISM: Train–evaluation matched cardinal-translation augmentation

HYPOTHESIS: Sampling only the centered and four one-pixel cardinal translations during training will exceed 9,188 correct predictions by concentrating the fixed exposure on the spatial views that improved evaluation performance.

INTENDED_EDIT: Replace uniform sampling across 25 translations up to two pixels with uniform sampling across the five centered/cardinal one-pixel views used by evaluation; retain horizontal flips and all other training settings.

EVIDENCE: Cardinal translation-flip averaging improved validation correct from 9,138 to 9,167, while adding diagonal views reduced it to 9,166; the batch-size-48 experiment also failed to improve 9,188, motivating a train–evaluation augmentation alignment rather than another step-count change.

<<<<<<< SEARCH
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    windows = padded.unfold(2, 28, 1).unfold(3, 28, 1)
    batch_index = torch.arange(images.shape[0], device=images.device)
    shifts = torch.randint(0, 5, (images.shape[0], 2), device=images.device)
    images = windows[batch_index, :, shifts[:, 0], shifts[:, 1]]
=======
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    windows = padded.unfold(2, 28, 1).unfold(3, 28, 1)
    batch_index = torch.arange(images.shape[0], device=images.device)
    cardinal_offsets = torch.tensor(
        ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)),
        device=images.device,
    )
    choices = torch.randint(
        0, cardinal_offsets.shape[0], (images.shape[0],), device=images.device
    )
    shifts = cardinal_offsets[choices]
    images = windows[batch_index, :, shifts[:, 0], shifts[:, 1]]
>>>>>>> REPLACE