MECHANISM: Augmentation-matched one-pixel translation training

HYPOTHESIS: Restricting training translations from ±2 pixels to ±1 will exceed 9,162 correct predictions by matching the successful validation views and reducing exposure spent on strongly displaced images.

INTENDED_EDIT: Change random training crops from the five-by-five ±2 translation grid to the three-by-three ±1 grid; retain flips, optimization, architecture, and center-weighted inference.

EVIDENCE: One-pixel cardinal inference improved correctness from 9,134 to 9,159, while adding more displaced inference influence regressed and center weighting improved the result to 9,162; this suggests moderate translation robustness helps but excessive displacement is harmful.

<<<<<<< SEARCH
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    windows = padded.unfold(2, 28, 1).unfold(3, 28, 1)
    offsets = torch.randint(0, 5, (batch_size, 2), device=images.device)
=======
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    windows = padded.unfold(2, 28, 1).unfold(3, 28, 1)
    offsets = torch.randint(0, 3, (batch_size, 2), device=images.device)
>>>>>>> REPLACE