MECHANISM: Modestly increased cardinal-shift exposure

HYPOTHESIS: Reducing centered training crops from 25% to approximately 22.7% while distributing the difference across balanced cardinal shifts will exceed 9,206 correct predictions.

INTENDED_EDIT: Deterministically convert one-eleventh of centered candidates into step-rotated one-pixel cardinal crops, preserving the existing diagonal cardinalization and computational profile.

EVIDENCE: Increasing center exposure from 25% to 27.3% reduced correctness from 9,206 to 9,198, while replacing diagonal crops with cardinal shifts previously improved correctness from 9,196 to 9,206; a symmetric, modest shift away from center is therefore the most directly supported next test.

<<<<<<< SEARCH
    del step, total_steps
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    batch_indices = torch.arange(images.shape[0], device=images.device)
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
    diagonal_mask = (offset_y != 2) & (offset_x != 2)
    center_x_mask = diagonal_mask & ((batch_indices & 1) == 0)
    offset_x.masked_fill_(center_x_mask, 2)
    offset_y.masked_fill_(diagonal_mask & ~center_x_mask, 2)
=======
    del total_steps
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    batch_indices = torch.arange(images.shape[0], device=images.device)
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
    diagonal_mask = (offset_y != 2) & (offset_x != 2)
    center_x_mask = diagonal_mask & ((batch_indices & 1) == 0)
    offset_x.masked_fill_(center_x_mask, 2)
    offset_y.masked_fill_(diagonal_mask & ~center_x_mask, 2)

    cycle = (batch_indices + step) % 44
    promoted_center = (
        (offset_y == 2)
        & (offset_x == 2)
        & ((cycle % 11) == 0)
    )
    direction = cycle // 11
    offset_y.masked_fill_(promoted_center & (direction == 0), 1)
    offset_y.masked_fill_(promoted_center & (direction == 1), 3)
    offset_x.masked_fill_(promoted_center & (direction == 2), 1)
    offset_x.masked_fill_(promoted_center & (direction == 3), 3)
>>>>>>> REPLACE