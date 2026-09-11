MECHANISM: Center-and-cardinal translation augmentation

HYPOTHESIS: Replacing diagonal training shifts with center/cardinal ±1 shifts weighted like the successful validation ensemble will exceed 9,196 correct predictions by retaining beneficial translation invariance while eliminating unsupported diagonal transforms.

INTENDED_EDIT: Sample only the centered crop and four one-pixel cardinal translations, with probabilities proportional to validation weights 3:2:2:2:2; retain flips and all other training settings.

EVIDENCE: Restricting translations from ±2 to ±1 improved correctness from 9,166 to 9,196, while adding diagonal validation views previously reduced correctness to 9,155; this directly motivates removing the remaining diagonal training transforms.

<<<<<<< SEARCH
    offset_y = (
        torch.randint(0, 2, (images.shape[0],), device=images.device)
        + torch.randint(0, 2, (images.shape[0],), device=images.device)
        + 1
    )
    offset_x = (
        torch.randint(0, 2, (images.shape[0],), device=images.device)
        + torch.randint(0, 2, (images.shape[0],), device=images.device)
        + 1
    )
=======
    transform_draw = torch.randint(
        0, 11, (images.shape[0],), device=images.device
    )
    transform_index = torch.where(
        transform_draw < 3,
        torch.zeros_like(transform_draw),
        1 + (transform_draw - 3) // 2,
    )
    offsets = torch.tensor(
        [[2, 2], [1, 2], [3, 2], [2, 1], [2, 3]],
        device=images.device,
    )[transform_index]
    offset_y = offsets[:, 0]
    offset_x = offsets[:, 1]
>>>>>>> REPLACE