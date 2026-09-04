MECHANISM: Center-biased cardinal translation augmentation

HYPOTHESIS: Sampling the native centered view 50% of the time while retaining all four one-pixel cardinal shifts will exceed 9,282 correct predictions by improving clean-image learning during the fixed two-pass exposure without discarding the translation robustness responsible for the 9,262-result gain.

INTENDED_EDIT: Retain the verified architecture, inference ensemble, schedule, and tail averaging, but change training-position probabilities from 20% each to 50% centered and 12.5% for each cardinal shift.

EVIDENCE: Restricting augmentation to centered/cardinal positions raised correctness from 9,209 to 9,262, and tail averaging then raised it to 9,282; because validation images retain their native alignment and exposure is limited, biasing that proven augmentation set toward the centered view is the most direct low-cost refinement.

<<<<<<< SEARCH
    positions = torch.randint(0, 5, (batch,), device=images.device)
    row_offsets = torch.tensor(
        (1, 0, 2, 1, 1), device=images.device
    )[positions].unsqueeze(1)
    col_offsets = torch.tensor(
        (1, 1, 1, 0, 2), device=images.device
    )[positions].unsqueeze(1)
=======
    positions = torch.randint(0, 8, (batch,), device=images.device)
    row_offsets = torch.tensor(
        (1, 1, 1, 1, 0, 2, 1, 1), device=images.device
    )[positions].unsqueeze(1)
    col_offsets = torch.tensor(
        (1, 1, 1, 1, 1, 1, 0, 2), device=images.device
    )[positions].unsqueeze(1)
>>>>>>> REPLACE