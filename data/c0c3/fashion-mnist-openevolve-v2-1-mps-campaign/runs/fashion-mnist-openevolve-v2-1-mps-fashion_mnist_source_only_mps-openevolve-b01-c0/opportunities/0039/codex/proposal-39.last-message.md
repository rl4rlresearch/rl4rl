MECHANISM: Center-weighted translation augmentation

HYPOTHESIS: Increasing centered training views from 20% to 33% will exceed 9,290 correct predictions by preserving centered-image detail while retaining balanced one-pixel translation robustness.

INTENDED_EDIT: Duplicate the centered offset in the augmentation choices, giving it twice the probability of each cardinal shift without changing compute or evaluation.

EVIDENCE: Removing centered training views reduced validation correct from 9,290 to 9,270, directly motivating a controlled increase in centered-view frequency while preserving all validated training components.

<<<<<<< SEARCH
    cardinal_offsets = torch.tensor(
        ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)),
        device=images.device,
    )
=======
    cardinal_offsets = torch.tensor(
        ((1, 1), (1, 1), (0, 1), (2, 1), (1, 0), (1, 2)),
        device=images.device,
    )
>>>>>>> REPLACE