MECHANISM: Late-state-biased temporal pooling

HYPOTHESIS: An 86-unit GRU processing frames 4–23 with a parameter-free 25% final-state residual will achieve at least 85% validation accuracy while retaining approximately 446,333,120 total MACs.

INTENDED_EDIT: Reduce the GRU to 86 units, use the best-performing tested 20-frame schedule, and bias classification toward the context-complete final recurrent state without adding learned operations.

EVIDENCE: The 86-unit frames 4–23 model missed qualification by only one validation example at 84.91%, while alternative 20-frame schedules performed worse; changing its zero-MAC temporal aggregation is therefore more informative than further schedule changes.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 91, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(91, 8)
=======
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(86, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 91, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 91, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 11:
            return list(range(available_frames))
        return list(range(4, available_frames - 6))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        pooled = 0.75 * mean_output + 0.25 * hidden[:, 0, :]
        return self.classifier(pooled)

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 13:
            return list(range(available_frames))
        return list(range(4, available_frames - 8))
>>>>>>> REPLACE