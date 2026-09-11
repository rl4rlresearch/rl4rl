MECHANISM: Selective high-band delta coarsening

HYPOTHESIS: A 71-unit GRU using seven adjacent-band delta pairs plus one pooled upper-band delta over the qualified 27-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 469,632,340 to approximately 464,945,275.

INTENDED_EDIT: Preserve the qualified recurrent width and temporal coverage while reducing delta features from nine to eight by pooling the three highest-frequency band pairs.

EVIDENCE: The 71-unit model remained qualified at 85.77% accuracy after reducing pooled deltas from ten to nine, whereas reducing recurrent width to 68 failed; this favors another conservative feature reduction while preserving recurrent capacity.

<<<<<<< SEARCH
        self.gru = nn.GRU(40, 75, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(150, 8)
=======
        self.gru = nn.GRU(28, 71, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(142, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 75, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 75, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 71, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 71, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        delta = (normalized - previous) * has_previous
        features = torch.cat((normalized, delta), dim=1)
=======
        delta = (normalized - previous) * has_previous
        paired_delta = 0.5 * (delta[:, 0::2] + delta[:, 1::2])
        pooled_delta = torch.cat(
            (
                paired_delta[:, :7],
                paired_delta[:, 7:].mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        features = torch.cat((normalized, pooled_delta), dim=1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = torch.cat((normalized, deltas), dim=2)
=======
        paired_deltas = 0.5 * (
            deltas[:, :, 0::2] + deltas[:, :, 1::2]
        )
        pooled_deltas = torch.cat(
            (
                paired_deltas[:, :, :7],
                paired_deltas[:, :, 7:].mean(dim=2, keepdim=True),
            ),
            dim=2,
        )
        features = torch.cat((normalized, pooled_deltas), dim=2)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        start = max(available_frames - 28, 0)
        return list(range(start, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(27, available_frames)
        start = max(available_frames - 28, 0)
        window_frames = available_frames - start
        if window_frames <= target_steps:
            return list(range(start, available_frames))
        return [
            start + round(index * (window_frames - 1) / (target_steps - 1))
            for index in range(target_steps)
        ]
>>>>>>> REPLACE