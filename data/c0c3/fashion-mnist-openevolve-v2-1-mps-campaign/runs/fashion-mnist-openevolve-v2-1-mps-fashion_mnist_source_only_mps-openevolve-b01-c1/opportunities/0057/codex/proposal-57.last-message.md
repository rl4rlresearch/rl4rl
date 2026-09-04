MECHANISM: Random crop microbatching

HYPOTHESIS: Independently sampling crop offsets for four groups within each batch will exceed 9,247 correct predictions by increasing stochastic crop diversity without adding model evaluations or the prohibitive overhead of per-example sampling.

INTENDED_EDIT: Split each padded batch into four contiguous groups, assign each group independent full and central crop offsets, and preserve the existing antithetic crops, flips, losses, architecture, and optimizer.

EVIDENCE: Per-example crop sampling timed out, while step-level deterministic offset balancing reached only 9,238 correct; inexpensive within-batch stochastic diversity tests the remaining middle ground while retaining the 9,247-correct curriculum.

<<<<<<< SEARCH
    offsets = torch.randint(0, 5, (2,))
    offset_y, offset_x = int(offsets[0]), int(offsets[1])
    central_offsets = torch.randint(1, 4, (2,))
    central_y, central_x = int(central_offsets[0]), int(central_offsets[1])
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    view_a = padded[
        :,
        :,
        offset_y : offset_y + 28,
        offset_x : offset_x + 28,
    ]
    view_b = padded[
        :,
        :,
        4 - offset_y : 4 - offset_y + 28,
        4 - offset_x : 4 - offset_x + 28,
    ]
    central_view = padded[
        :,
        :,
        central_y : central_y + 28,
        central_x : central_x + 28,
    ]
=======
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    padded_groups = padded.chunk(4, dim=0)
    offsets = torch.randint(0, 5, (len(padded_groups), 2))
    central_offsets = torch.randint(1, 4, (len(padded_groups), 2))
    view_a_groups = []
    view_b_groups = []
    central_groups = []
    for group_index, padded_group in enumerate(padded_groups):
        offset_y = int(offsets[group_index, 0])
        offset_x = int(offsets[group_index, 1])
        central_y = int(central_offsets[group_index, 0])
        central_x = int(central_offsets[group_index, 1])
        view_a_groups.append(
            padded_group[
                :,
                :,
                offset_y : offset_y + 28,
                offset_x : offset_x + 28,
            ]
        )
        view_b_groups.append(
            padded_group[
                :,
                :,
                4 - offset_y : 4 - offset_y + 28,
                4 - offset_x : 4 - offset_x + 28,
            ]
        )
        central_groups.append(
            padded_group[
                :,
                :,
                central_y : central_y + 28,
                central_x : central_x + 28,
            ]
        )
    view_a = torch.cat(view_a_groups, dim=0)
    view_b = torch.cat(view_b_groups, dim=0)
    central_view = torch.cat(central_groups, dim=0)
>>>>>>> REPLACE