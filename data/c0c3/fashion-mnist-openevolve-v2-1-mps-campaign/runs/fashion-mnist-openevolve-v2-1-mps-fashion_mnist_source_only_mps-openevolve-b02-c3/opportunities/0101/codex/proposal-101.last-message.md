MECHANISM: TTA-aligned anisotropic translation training

HYPOTHESIS: Matching training-crop frequencies to the best-verified inference weights will exceed 9,348 correct predictions by emphasizing the more reliable vertical translations without changing the model or evaluation ensemble.

INTENDED_EDIT: Replace symmetric 3:1:1:1:1 translation sampling with a 48:21:21:11:11 categorical distribution, the nearest integer representation of the current 3:1.3134765625:1.3134765625:0.6865234375:0.6865234375 TTA weights.

EVIDENCE: Vertical-favoring TTA improved correctness from 9,344 with symmetric weights to 9,348, while adaptive pooling and attention changes regressed; this directly motivates aligning the training exposure distribution with the successful anisotropy.

<<<<<<< SEARCH
    translation = torch.randint(0, 7, (batch,), device=images.device)
    offsets_y = torch.tensor(
        (1, 1, 1, 0, 2, 1, 1), device=images.device
    )[translation].unsqueeze(1)
    offsets_x = torch.tensor(
        (1, 1, 1, 1, 1, 0, 2), device=images.device
    )[translation].unsqueeze(1)
=======
    translation = torch.randint(0, 112, (batch,), device=images.device)
    offsets_y = (
        1
        - ((translation >= 48) & (translation < 69)).long()
        + ((translation >= 69) & (translation < 90)).long()
    ).unsqueeze(1)
    offsets_x = (
        1
        - ((translation >= 90) & (translation < 101)).long()
        + (translation >= 101).long()
    ).unsqueeze(1)
>>>>>>> REPLACE