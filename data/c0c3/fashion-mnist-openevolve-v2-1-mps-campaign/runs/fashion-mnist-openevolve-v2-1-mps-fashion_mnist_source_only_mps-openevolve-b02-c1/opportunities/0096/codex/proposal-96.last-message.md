MECHANISM: Normalization-aware background translation

HYPOTHESIS: Padding translated views with each image’s observed background level instead of normalized zero will exceed 9,286 correct predictions by removing artificial border strips from both training augmentation and TTA.

INTENDED_EDIT: Subtract each image’s minimum intensity before padding and restore it afterward, so translations use the image’s background value without assuming a particular normalization.

EVIDENCE: Temperature and TTA-weight tuning plateaued at 9,286 correct, while altered terminal pooling regressed to 9,234; this motivates correcting the existing augmentation itself without changing model capacity, view weights, or runtime materially.

<<<<<<< SEARCH
        padded = F.pad(images, (1, 1, 1, 1))
=======
        background = images.amin(dim=(2, 3), keepdim=True)
        padded = F.pad(images - background, (1, 1, 1, 1)) + background
>>>>>>> REPLACE

<<<<<<< SEARCH
    windows = F.pad(images, (1, 1, 1, 1)).unfold(2, 28, 1).unfold(3, 28, 1)
=======
    background = images.amin(dim=(2, 3), keepdim=True)
    padded = F.pad(images - background, (1, 1, 1, 1)) + background
    windows = padded.unfold(2, 28, 1).unfold(3, 28, 1)
>>>>>>> REPLACE