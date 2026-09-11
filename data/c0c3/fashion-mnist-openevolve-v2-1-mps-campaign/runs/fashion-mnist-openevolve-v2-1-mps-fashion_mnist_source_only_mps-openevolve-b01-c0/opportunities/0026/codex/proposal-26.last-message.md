MECHANISM: Shift-emphasized cardinal augmentation

HYPOTHESIS: Excluding the centered training view and sampling the four one-pixel cardinal translations uniformly will exceed 9,290 correct predictions by further strengthening the translation robustness favored by the existing evidence.

INTENDED_EDIT: Remove the centered offset from training augmentation while preserving evaluation views and all other settings.

EVIDENCE: Increasing the centered-view share from 20% to 50% reduced validation correct from 9,290 to 9,278, directly motivating a controlled change in the opposite direction.

<<<<<<< SEARCH
    cardinal_offsets = torch.tensor(
        ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)),
        device=images.device,
    )
=======
    cardinal_offsets = torch.tensor(
        ((0, 1), (2, 1), (1, 0), (1, 2)),
        device=images.device,
    )
>>>>>>> REPLACE