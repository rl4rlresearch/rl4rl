MECHANISM: Affine-free input normalization with GRU absorption

HYPOTHESIS: The qualified 110-unit, 22-step model will retain at least 85% validation accuracy when LayerNorm’s 40 affine parameters are removed, because the following GRU input weights and biases can represent the same scaling and offset, while inference MACs and recurrent steps remain unchanged and learned parameters decrease from 45,360 to 45,320.

INTENDED_EDIT: Restore the qualified 22-step, 110-unit, 219-coordinate dual-readout design and make its input LayerNorm non-affine.

EVIDENCE: Reference Design 2 achieved 85.15% accuracy at 770,624,880 MACs after one final-state coordinate was pruned; the failed second-coordinate pruning motivates preserving that qualified recurrent path and readout while reducing only functionally redundant normalization parameters.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.input_norm = nn.LayerNorm(20, elementwise_affine=False)
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(219, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(
            torch.cat((mean_output, hidden[:, 0, :-1]), dim=1)
        )

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(29, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if available_frames >= 6:
            return [
                frame
                for frame in schedule
                if frame
                not in (
                    0,
                    1,
                    2,
                    available_frames - 5,
                    available_frames - 4,
                    available_frames - 3,
                    available_frames - 2,
                )
            ]
        return schedule
>>>>>>> REPLACE