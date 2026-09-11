MECHANISM: Diagonal translation augmentation

HYPOTHESIS: Sampling diagonal one-pixel translations during training will exceed 9,345 correct predictions by improving local translation invariance without adding parameters, optimizer work, or material runtime.

INTENDED_EDIT: Expand the existing random-crop augmentation from six center/cardinal choices to ten choices comprising two centered crops and all eight one-pixel neighboring translations.

EVIDENCE: Raw-mean gating achieved the best 9,345-correct result, while multiple added representation paths regressed or timed out; the current augmentation covers only center and cardinal translations, so diagonal jitter tests an untried generalization axis at essentially unchanged cost.

<<<<<<< SEARCH
    view_ids = torch.randint(0, 6, (batch_size,), device=images.device)
    offset_table = torch.tensor(
        ((1, 1), (1, 1), (0, 1), (2, 1), (1, 0), (1, 2)),
        device=images.device,
    )
=======
    view_ids = torch.randint(0, 10, (batch_size,), device=images.device)
    offset_table = torch.tensor(
        (
            (1, 1), (1, 1),
            (0, 0), (0, 1), (0, 2),
            (1, 0),         (1, 2),
            (2, 0), (2, 1), (2, 2),
        ),
        device=images.device,
    )
>>>>>>> REPLACE