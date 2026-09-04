MECHANISM: Deterministic validation-matched cardinal augmentation cycle

HYPOTHESIS: Cycling through center and cardinal translations in the exact 3:2:2:2:2 validation-view ratio will exceed 9,206 correct predictions by reducing per-batch augmentation sampling noise during the fixed two-pass exposure.

INTENDED_EDIT: Replace random triangular shift generation and diagonal cardinalization with a low-overhead deterministic 11-example cycle containing three centered crops and two crops for each one-pixel cardinal translation.

EVIDENCE: Removing diagonal translations improved correctness from 9,196 to 9,206; the earlier validation-matched 3:2:2:2:2 proposal timed out, while deterministic balanced cardinalization completed successfully, motivating a deterministic formulation of the previously unverified distribution.

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
    phase = (step * BATCH_SIZE + batch_indices) % 11
    offset_y = torch.full_like(batch_indices, 2)
    offset_x = torch.full_like(batch_indices, 2)
    offset_y.masked_fill_((phase >= 3) & (phase < 5), 1)
    offset_y.masked_fill_((phase >= 5) & (phase < 7), 3)
    offset_x.masked_fill_((phase >= 7) & (phase < 9), 1)
    offset_x.masked_fill_(phase >= 9, 3)
>>>>>>> REPLACE