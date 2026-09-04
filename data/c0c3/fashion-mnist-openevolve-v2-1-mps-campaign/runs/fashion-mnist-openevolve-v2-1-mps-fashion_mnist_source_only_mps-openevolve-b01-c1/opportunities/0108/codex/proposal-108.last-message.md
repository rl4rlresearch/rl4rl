MECHANISM: Per-example stratified translation augmentation

HYPOTHESIS: Sampling crop offsets independently for every image will exceed 9,257 correct predictions by increasing translation diversity within each optimizer step while preserving the proven augmentation range, architecture, and loss.

INTENDED_EDIT: Replace batch-shared translation offsets with independently sampled per-image offsets for both full-range and central crops, using tensor gathers to construct each view.

EVIDENCE: Center-weighted crop evaluation improved the available design, demonstrating useful translation sensitivity, while stronger rotation and altered pooling regressed; increasing diversity within the existing successful translation augmentation tests shift robustness without introducing a new transformation or parameters.

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
    batch_count = images.shape[0]
    offsets = torch.randint(
        0, 5, (batch_count, 2), device=images.device
    )
    central_offsets = torch.randint(
        1, 4, (batch_count, 2), device=images.device
    )
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    coordinates = torch.arange(28, device=images.device)

    def shifted_view(
        offset_y: torch.Tensor,
        offset_x: torch.Tensor,
    ) -> torch.Tensor:
        row_index = (
            offset_y[:, None] + coordinates[None, :]
        )[:, None, :, None]
        rows = torch.gather(
            padded,
            2,
            row_index.expand(
                -1, padded.shape[1], -1, padded.shape[3]
            ),
        )
        column_index = (
            offset_x[:, None] + coordinates[None, :]
        )[:, None, None, :]
        return torch.gather(
            rows,
            3,
            column_index.expand(
                -1, rows.shape[1], rows.shape[2], -1
            ),
        )

    view_a = shifted_view(offsets[:, 0], offsets[:, 1])
    view_b = shifted_view(4 - offsets[:, 0], 4 - offsets[:, 1])
    central_view = shifted_view(
        central_offsets[:, 0], central_offsets[:, 1]
    )
>>>>>>> REPLACE