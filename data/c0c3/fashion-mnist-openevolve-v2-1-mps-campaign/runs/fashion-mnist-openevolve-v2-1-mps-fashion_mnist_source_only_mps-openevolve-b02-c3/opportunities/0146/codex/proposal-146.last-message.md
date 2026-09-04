MECHANISM: Reverse-anisotropic translation sampling

HYPOTHESIS: Sampling translations in a 9:2:2:4:4 center/up/down/left/right ratio will exceed 9,348 correct predictions by reversing the vertical oversampling that previously reduced correctness.

INTENDED_EDIT: Keep the center-translation probability unchanged while transferring training exposure from vertical shifts to horizontal shifts; architecture, evaluation ensemble, and calibration remain unchanged.

EVIDENCE: Changing the original scaled 9:3:3:3:3 distribution to 9:4:4:2:2 reduced validation correctness from 9,348 to 9,338. Because the center probability was identical, this supplies directional evidence that favoring vertical over horizontal shifts is harmful and motivates testing the symmetric reverse allocation.

<<<<<<< SEARCH
    translation = torch.randint(0, 7, (batch,), device=images.device)
    offsets_y = torch.tensor(
        (1, 1, 1, 0, 2, 1, 1), device=images.device
    )[translation].unsqueeze(1)
    offsets_x = torch.tensor(
        (1, 1, 1, 1, 1, 0, 2), device=images.device
    )[translation].unsqueeze(1)
=======
    translation = torch.randint(0, 21, (batch,), device=images.device)
    offsets_y = torch.tensor(
        (
            1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 0,
            2, 2,
            1, 1, 1, 1,
            1, 1, 1, 1,
        ),
        device=images.device,
    )[translation].unsqueeze(1)
    offsets_x = torch.tensor(
        (
            1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1,
            1, 1,
            0, 0, 0, 0,
            2, 2, 2, 2,
        ),
        device=images.device,
    )[translation].unsqueeze(1)
>>>>>>> REPLACE