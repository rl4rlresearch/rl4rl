MECHANISM: Evaluation-aligned anisotropic translation sampling

HYPOTHESIS: Sampling training translations in the 9:4:4:2:2 ratio implied by the best TTA weights will exceed 9,348 correct predictions by improving invariance to the more valuable vertical shifts while reducing exposure to less reliable horizontal shifts.

INTENDED_EDIT: Replace the current 3:1:1:1:1 translation distribution with a 21-way discrete approximation of the verified evaluation weights, leaving architecture, runtime, and example accounting unchanged.

EVIDENCE: Axial translation TTA raised correctness from 9,344 to 9,348, and its best weights scale almost exactly to 9:4:4:2:2; diagonal translations subsequently reduced correctness to 9,335, showing that matching the useful translation directions is consequential.

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
            0, 0, 0, 0,
            2, 2, 2, 2,
            1, 1, 1, 1,
        ),
        device=images.device,
    )[translation].unsqueeze(1)
    offsets_x = torch.tensor(
        (
            1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            0, 0, 2, 2,
        ),
        device=images.device,
    )[translation].unsqueeze(1)
>>>>>>> REPLACE